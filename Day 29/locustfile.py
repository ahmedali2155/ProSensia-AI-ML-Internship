from locust import HttpUser, task, between


class TitanicUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        payload = {
            "Pclass": 3,
            "Sex": "male",
            "Age": 22,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked": "S"
        }

        self.client.post(
            "/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )