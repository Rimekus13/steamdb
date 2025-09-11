# run_pipeline.py
import argparse
from etl.config import Config
from etl.bronze_extract import extract_app
from etl.silver_clean import to_silver
from etl.gold_build import build_gold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "incr"], default="incr")
    args = parser.parse_args()

    if not Config.app_ids:
        raise SystemExit("❌ Aucun APP_ID configuré dans .env (clé APP_IDS).")

    dts = set()
    for app_id in Config.app_ids:
        print(f"\n🚀 [STEP 1] Extraction RAW → datalake pour app_id={app_id} ...")
        dt = extract_app(app_id, mode=args.mode, for_airflow=False)
        print(f"✅ RAW datalake OK : data/raw/{app_id}/{dt}")
        dts.add((app_id, dt))

    for app_id, dt in dts:
        print(f"\n🚀 [STEP 2] Transformation Silver → Mongo (clean) pour app_id={app_id}, date={dt} ...")
        to_silver(app_id, dt=dt, for_airflow=False)
        print(f"✅ Silver in Mongo OK (collection reviews_clean)")

    print("\n🚀 [STEP 3] Construction Gold (agrégats / cooccurrences) ...")
    build_gold(for_airflow=False)
    print("✅ Gold in Mongo OK (collections cooccurrences_counts / cooccurrences_percent)")

    print("\n🎉 Pipeline terminé avec succès !")

if __name__ == "__main__":
    main()
