# dags/steam_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.config import Config
from etl.bronze_extract import extract_app
from etl.silver_clean import to_silver
from etl.gold_build import build_gold

default_args = {
    "owner": "steam_etl",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="steam_etl_gcp",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["steam", "etl", "gcp"],
) as dag:

    def extract():
        print("[DEBUG] app_ids:", Config.app_ids)
        print("[DEBUG] bucket:", Config.gcs_bucket)
        for app_id in Config.app_ids:
            # ajuste pages/per_page si besoin
            extract_app(app_id, mode="full", pages=50, per_page=100)

    def silver():
        dt = datetime.utcnow().strftime("%Y-%m-%d")
        for app_id in Config.app_ids:
            to_silver(app_id, dt=dt)

    def gold():
        # traite tous les app_id présents en SILVER
        build_gold()

    t_extract = PythonOperator(
        task_id="extract_raw",
        python_callable=extract,
    )

    t_silver = PythonOperator(
        task_id="build_silver",
        python_callable=silver,
    )

    t_gold = PythonOperator(
        task_id="build_gold",
        python_callable=gold,
    )

    # ==============================
    # Option B : déployer Streamlit
    # ==============================
    # Ce task essaie d'utiliser docker compose depuis le conteneur Airflow.
    # Pré-requis si tu veux que ça marche :
    #   - socket docker monté dans le conteneur: -v /var/run/docker.sock:/var/run/docker.sock
    #   - binaire docker + plugin compose disponibles dans le conteneur
    # Si non dispo, le task log un WARN et s'arrête (sans faire échouer le DAG).
    deploy_streamlit = BashOperator(
        task_id="deploy_streamlit",
        bash_command=r"""
            set -euo pipefail
            echo "[INFO] Deploy Streamlit..."

            # Dossier où se trouve ton docker-compose.yml côté conteneur Airflow.
            # Ajuste si nécessaire (par ex. /opt/airflow si tu montes le dépôt ici).
            APP_DIR="/opt/airflow"
            cd "$APP_DIR"

            # Choix de la commande compose
            if docker compose version >/dev/null 2>&1; then
              DC="docker compose"
            elif command -v docker-compose >/dev/null 2>&1; then
              DC="docker-compose"
            else
              echo "[WARN] docker compose non disponible dans le conteneur Airflow. Skip déploiement Streamlit."
              exit 0
            fi

            # Build & (re)start uniquement le service streamlit
            $DC pull streamlit || true
            $DC up -d --build streamlit
            $DC ps
            echo "[OK] Streamlit déployé (ou déjà en cours d'exécution)."
        """,
        trigger_rule="all_success",  # ne lance que si Gold a réussi
    )

    t_extract >> t_silver >> t_gold >> deploy_streamlit
