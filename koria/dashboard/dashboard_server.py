from flask import Flask, request, jsonify

class DashboardServer:
    def __init__(self, tasks):
        self.tasks = tasks
        self.app = Flask(__name__)

        @self.app.route('/ner')
        def ner():
            if request.method == "GET":
                sentence = request.args["sentence"]

                return jsonify({"answer": self.tasks["ner"].predict(sentence=sentence)})
                
        @self.app.route('/mrc')
        def mrc():
            if request.method == "GET":
                sentence = request.args["sentence"]
                question = request.args["question"]
                
                return jsonify({"answer": self.tasks["mrc"].predict(sentence=sentence, question=question)})

        @self.app.route('/qg')
        def qg():
            return ""

    def run(self):
        self.app.run(host="localhost",port=5001)
    