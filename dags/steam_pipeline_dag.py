from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.config import Config
from etl.bronze_extract import extract_app
from etl.silver_clean import to_silver
from etl.gold_build import build_gold

default_args = {"owner": "steam_etl", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="steam_etl_gcp",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["steam","etl","gcp"],
) as dag:

    def task_extract():
        for app_id in Config.app_ids:
            extract_app(app_id, mode="incr", for_airflow=True)

    def task_silver():
        dt = datetime.utcnow().strftime("%Y-%m-%d")
        for app_id in Config.app_ids:
            to_silver(app_id, dt=dt, for_airflow=True)

    def task_gold():
        build_gold(for_airflow=True)

    t_extract = PythonOperator(task_id="extract_raw", python_callable=task_extract)
    t_silver  = PythonOperator(task_id="build_silver", python_callable=task_silver)
    t_gold    = PythonOperator(task_id="build_gold", python_callable=task_gold)

    t_extract >> t_silver >> t_gold
