import re
import unittest
import uuid

from app import app


class LoanAppSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _csrf_from(self, path):
        response = self.client.get(path)
        html = response.get_data(as_text=True)
        match = re.search(r'name="_csrf_token" value="([^"]+)"', html)
        self.assertIsNotNone(match)
        return match.group(1)

    def test_auth_and_loan_flow(self):
        token = self._csrf_from("/register")
        email = f"test_{uuid.uuid4().hex[:8]}@example.com"
        password = "Pass12345"

        register = self.client.post(
            "/register",
            data={
                "_csrf_token": token,
                "name": "Smoke User",
                "email": email,
                "role": "user",
                "password": password,
            },
            follow_redirects=False,
        )
        self.assertEqual(register.status_code, 302)

        token = self._csrf_from("/login")
        login = self.client.post(
            "/login",
            data={"_csrf_token": token, "email": email, "password": password},
            follow_redirects=False,
        )
        self.assertEqual(login.status_code, 302)

        predict = self.client.get("/predict")
        self.assertEqual(predict.status_code, 200)


if __name__ == "__main__":
    unittest.main()
