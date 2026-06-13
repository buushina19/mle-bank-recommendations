import requests

BASE_URL = "http://127.0.0.1:8000"

payload = {
    "ncodpers": 123456,
    "ind_empleado": "N",
    "sexo": "H",
    "age": 45,
    "ind_nuevo": 0,
    "antiguedad": 150,
    "indrel_1mes": "1",
    "tiprel_1mes": "A",
    "indresi": 1,
    "indext": 0,
    "conyuemp": 0,
    "canal_entrada": "KFC",
    "ind_actividad_cliente": 1,
    "renta": 120000,
    "segmento": "02",
    "products": {
        "ind_cco_fin_ult1": 1,
        "ind_tjcr_fin_ult1": 0,
        "ind_recibo_ult1": 1,
    },
}


def test_health():
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    response.raise_for_status()
    assert response.json()["status"] == "healthy"


def test_recommend():
    response = requests.post(f"{BASE_URL}/recommend", json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    assert data["ncodpers"] == payload["ncodpers"]
    assert len(data["recommendations"]) == 7
    print("Recommendations:")
    for item in data["recommendations"]:
        print(f"  {item['product_code']} ({item['product_name']}): {item['score']:.4f}")


def test_metrics():
    response = requests.get(f"{BASE_URL}/metrics", timeout=10)
    response.raise_for_status()
    assert "recommendation_requests_total" in response.text


if __name__ == "__main__":
    test_health()
    test_recommend()
    test_metrics()
    print("All tests passed")
