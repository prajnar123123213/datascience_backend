# api/chart.py

import csv
import os
from flask import Blueprint, jsonify
from flask_restful import Api, Resource

chart_api = Blueprint('chart_api', __name__, url_prefix='/api/chart')
api = Api(chart_api)

CSV_FILE = 'chart.csv'

class ChartAPI:
    class _Definitions(Resource):
        def get(self):
            definitions = []
            if not os.path.exists(CSV_FILE):
                return jsonify({"error": "CSV file not found."}), 404

            with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    definitions.append({
                        "answer": row.get("answer", "").strip(),
                        "description": row.get("description", "").strip()
                    })
            return jsonify(definitions)

    api.add_resource(_Definitions, '/definitions')
