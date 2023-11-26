import time

import joblib
import requests
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)


def get_user(id_user: int):  # (черный ящик) эмулируем запрос в реестр кредитных сведений о клиенте
    url = "http://151.0.50.17:25564/"
    data = {"id": id_user}
    return requests.post(url, json=data)


@app.route('/predict', methods=['POST'])
def predict():
    a = time.time()
    data = request.get_json()
    user_data = get_user(data["id"])
    df = pd.DataFrame([user_data.json()["user"]])
    df = df[user_data.json()["data"]]
    surplus = data["money"] / (df['monthly_inhand_salary'][0] * 88.88)
    result_money_maximum = (df['monthly_inhand_salary'][0] * 88.88) * 0.45
    y_pred_proba = loaded_model.predict_proba(df)
    print(y_pred_proba[:, 1])
    user = user_data.json()
    tmp = {key: value for key, value in user['user'].items() if key[:11] == 'occupation_'}
    occupation = next(key.split('_')[1] for key, value in tmp.items() if value == 1.0)

    filtered_user_data = {key: value for key, value in user['user'].items() if key[:11] != 'occupation_'}
    filtered_user_data["occupation"] = occupation

    tmp = {key: value for key, value in filtered_user_data.items() if key[:17] == 'payment_behaviour'}
    payment_behaviour = next((key.split('_')[2:] for key, value in tmp.items() if value == 1.0),
                             ["High", "spent", "Large", "value", "payments"])

    filtered_user_data = {key: value for key, value in filtered_user_data.items() if key[:17] != 'payment_behaviour'}
    filtered_user_data["payment_behaviour"] = '_'.join(payment_behaviour)

    recommendation = []
    if filtered_user_data['num_of_loan'] > 3:
        recommendation.append("Вам необходимо уменьшить количество действующих кредитов")
    if filtered_user_data['amount_invested_monthly'] < 45:
        recommendation.append("Вам необходимо больше инвестировать")

    print(time.time() - a)
    return jsonify({
        "result_score": float(y_pred_proba[:, 1]),
        "result_money": 1 if surplus > 0.45 else 0,
        "result_money_maximum": result_money_maximum,
        "user_data": filtered_user_data,
        "recommendation": recommendation,
    })


@app.route('/calc', methods=['POST'])
def calc():
    data = request.get_json()
    sum = data["sum"]
    year = data["year"]
    stavka = data["stavka"]
    stavkaMonth = stavka / 12 / 100
    obshStavka = (1 + stavkaMonth) ** (year * 12)
    itogo = round(sum * stavkaMonth * obshStavka / (obshStavka - 1), 2)
    return jsonify({'result': itogo})


if __name__ == '__main__':
    loaded_model = joblib.load('GradientBoosting_auc-roc-0.9466.joblib')
    app.run(host='0.0.0.0', port=25565)
