import api.card_detection as cd
class Image:
    def __init__(self,url):
        self.url = url

    def ret(self):
        answer = cd.get_image(self.url)
        return answer
