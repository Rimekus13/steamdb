# dags/steam_pipeline_dag.py
from datetime import datetime, timedelta
import sys, pathlib

from airflow import DAG
from airflow.operators.python import PythonOperator, get_current_context

# ====== Import local package etl/* ======
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.config import Config
from etl.bronze_extract import extract_app
from etl.silver_clean import to_silver
from etl.gold_build import build_gold
from etl.mongo_utils import ensure_indexes

default_args = {
    "owner": "steam_etl",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="steam_etl_mongo_local",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",           # passe à "*/30 * * * *" si tu veux du micro-batch
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["steam", "etl", "mongo"],
) as dag:

    def task_ensure_indexes():
        # crée les index RAW/SILVER/GOLD pour chaque app_id (idempotent)
        for app_id in Config.app_ids:
            ensure_indexes(app_id, for_airflow=True)

    def task_extract():
        """
        Lance l'extract RAW pour chaque app_id.
        Pousse via XCom la date 'dt' utilisée (AAAA-MM-JJ) pour aligner Silver.
        """
        ctx = get_current_context()
        ti = ctx["ti"]

        run_dt = None
        for app_id in Config.app_ids:
            dt = extract_app(app_id, mode="incr", for_airflow=True)
            # la première valeur suffit (toutes seront identiques à l'UTC près)
            if run_dt is None:
                run_dt = dt

        if run_dt is None:
            # filet de sécurité
            run_dt = datetime.utcnow().strftime("%Y-%m-%d")

        ti.xcom_push(key="run_dt", value=run_dt)
        print(f"[XCOM] Pushed run_dt={run_dt}")

    def task_silver():
        """
        Lit la date 'dt' poussée par extract pour ouvrir le bon dossier RAW.
        """
        ctx = get_current_context()
        ti = ctx["ti"]
        dt = ti.xcom_pull(task_ids="extract_raw", key="run_dt") or datetime.utcnow().strftime("%Y-%m-%d")

        for app_id in Config.app_ids:
            to_silver(app_id, dt=dt, for_airflow=True)

    def task_gold():
        build_gold(for_airflow=True)

    t_idx     = PythonOperator(task_id="ensure_indexes", python_callable=task_ensure_indexes)
    t_extract = PythonOperator(task_id="extract_raw",    python_callable=task_extract)
    t_silver  = PythonOperator(task_id="build_silver",   python_callable=task_silver)
    t_gold    = PythonOperator(task_id="build_gold",     python_callable=task_gold)

    t_idx >> t_extract >> t_silver >> t_gold
