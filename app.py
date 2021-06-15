from flask import Flask, request
from flask_cors import CORS
import base64
from nst import *

app = Flask(__name__)
CORS(app)

@app.route('/',  methods=['POST'])
def NST_reqeust():
    request_data = request.get_json(silent=True)
    style_img = base64.b64decode(str(request_data["style"]))
    content_img = base64.b64decode(str(request_data["content"]))
    return NST(style_img, content_img)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
