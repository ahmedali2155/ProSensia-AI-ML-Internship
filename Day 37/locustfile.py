from locust import HttpUser, task, between


class TitanicUser(HttpUser):
    wait_time = between(1, 2)

    @task(8)
    def predict(self):
        payload = {
            "Pclass": 1,
            "Sex": "male",
            "Age": 70,
            "SibSp": 5,
            "Parch": 5,
            "Fare": 250,
            "Embarked": "S"
        }

        self.client.post("/predict", json=payload)

    @task(1)
    def drift_metrics(self):
        self.client.get("/metrics/drift")

    @task(1)
    def ab_metrics(self):
        self.client.get("/ab/metrics")