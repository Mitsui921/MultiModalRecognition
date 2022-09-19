import json
import requests
import re
from requests.auth import HTTPDigestAuth


class HttpRequest(object):
    """不记录任何的请求方法"""

    @classmethod
    def request(cls, method, url, data, auth=None, headers=None):  # 这里分别需要传人
        method = method.upper()  # 这里将传入的请求方法统一大写，然后进行判断采用什么方法
        if method == 'POST':
            return requests.post(url=url, data=data, auth=auth, headers=headers)
        elif method == 'GET':
            return requests.get(url=url, params=data, auth=auth, headers=headers)
        return f"目前没有{method}请求方法，只有POST和Get请求方法！"


if __name__ == '__main__':
    http = HttpRequest()
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
    }
    auth = HTTPDigestAuth("Default User", "robotics")
    data = {
        "regain": "continue", "execmode": "continue", "cycle": "forever", "condition": "none",
        "stopatbp": "disabled", "alltaskbytsp": "false"
    }
    # 生成物料 -> 1
    url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/nCount?json=1'
    response = http.request(method='get', url=url, data="", auth=auth, headers=headers)
    if (response.status_code == 200):
        json_data = json.loads(response.text)["_embedded"]["_state"][0]["value"]
        print(json_data)
    else:
        print("访问失败")

    # 放置物料的位置 -> 2
    url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/number_count?action=set&json=1'
    data = {
        "value": '5'
    }
    response = http.request(method='post', url=url, data=data, auth=auth, headers=headers)
    if (response.status_code == 204):

        print("正常")
    else:
        print("访问失败")

    # 获取物料坐标 -> 3
    url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/PosXY?json=1'

    # response = http.request(method='post', url=url, data=data, auth=auth)
    response = http.request(method='get', url=url, data="", auth=auth, headers=headers)
    if(response.status_code == 200):
        json_data = json.loads(response.text)["_embedded"]["_state"][0]["value"]
        print(json_data)
    else:
        print("访问失败")

    # 取物料
    url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/PosXY?action=set&json=1'
    if not json_data:
        print(json_data)
    data = {
        "value": str(json_data)
    }
    # response = http.request(method='post', url=url, data=data, auth=auth)
    response = http.request(method='post', url=url, data=data, auth=auth, headers=headers)
    if (response.status_code == 204):

        print("修改数据成功：")
    else:
        print("访问失败")

    # 放物料
    url = 'https://577p16s514.oicp.vip/rw/iosystem/signals/do_test1?action=set'

    # start
    data = {
        "lvalue": '1'
    }
    response = http.request(method='post', url=url, data=data, auth=auth, headers=headers)
    if (response.status_code == 204):
        # json_data = json.loads(response.text)["_embedded"]["_state"][0]["value"]
        print("修改IO信号成功：")
    else:
        print("访问失败")


