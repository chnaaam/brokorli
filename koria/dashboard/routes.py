from flask import Blueprint, request, jsonify

def routes(url_prefix, data_manager, auto_labeling_manager):
    bp = Blueprint("auto_labeling", __name__, url_prefix=url_prefix)

    @bp.route("/ner", methods=["POST"])
    def entity_labeling():
        if request.method == "POST":
            file_name = request.json["fileName"]
            sidx = request.json["sidx"]

            data = auto_labeling_manager.entity_labeling(data_manager.get_data(file_name), sidx)

            return jsonify(data)

    return bp