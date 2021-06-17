from flask import Flask, request
from flask_cors import CORS, cross_origin

import base64
from nst import *

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/',  methods=['POST'])
@cross_origin()
def NST_reqeust():
    request_data = request.get_json(silent=True)
    style_img = base64.b64decode(str(request_data["style"]))
    content_img = base64.b64decode(str(request_data["content"]))
    return NST(style_img, content_img)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
