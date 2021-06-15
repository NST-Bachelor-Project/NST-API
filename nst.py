import base64

def NST(style_img, content_img):
    print(type(style_img))
    result = {}
    result["image"] = 'data:image/jpg;base64,' + base64.b64encode(style_img).decode('utf-8')
    return result