import requests
import pandas as pd

API_URL = "https://anomalydetection-api.onrender.com/docs"

def test_api_with_csv():
    df = pd.read_csv("data/test_data.csv")

    success = 0
    failed = 0

    for i, row in df.iterrows():
        # Remove target column
        data = row.drop("Class").to_dict()

        try:
            response = requests.post(API_URL, json=data)

            if response.status_code == 200:
                success += 1
            else:
                failed += 1
                print(f"Failed at row {i}: {response.text}")

        except Exception as e:
            failed += 1
            print(f"Error at row {i}: {e}")

    print(f"Success: {success}")
    print(f"Failed: {failed}")

    # Fail test if too many errors
    assert failed == 0