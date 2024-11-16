import time
import traceback

import zerohertzLib as zz
from selenium import webdriver

START = "https://zerohertz.github.io/aws-neuron-sdk-aws-inferentia/"


class Browser:
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
        )
        self.browser = webdriver.Chrome(options)
        self.browser.get("https://github.com/login")
        while self.browser.current_url != "https://github.com/":
            time.sleep(0.1)
        self.browser.get(START)
        ready = input("READY?:\t")
        if ready.lower() != "y":
            exit()

    def next_post(self):
        self.browser.switch_to.default_content()
        _nxt = self.browser.find_element(
            "xpath", "/html/body/main/div[2]/div[1]/article/footer/div[2]/div[2]/a"
        )
        nxt = _nxt.text
        self.browser.get(_nxt.get_attribute("href"))
        return nxt

    def __call__(self):
        time.sleep(2)
        tmp = self.browser.find_element(
            "xpath", "/html/body/main/div[2]/div[1]/article/header/h1"
        ).text
        self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        return tmp, self.next_post()


def main():
    logger = zz.logging.Logger("GITALK")
    try:
        browser = Browser()
        logger.info("login completed")
        while True:
            tmp, nxt = browser()
            logger.info(f"tmp: {tmp}")
            logger.info(f"nxt: {nxt}")
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())

        import pdb

        pdb.set_trace()

        time.sleep(1000000)


if __name__ == "__main__":
    main()
