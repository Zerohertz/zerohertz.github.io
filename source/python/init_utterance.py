import time
import traceback

import zerohertzLib as zz
from selenium import webdriver

# https://github.com/user-attachments/assets/3b336180-984a-4ead-a8cf-4e45317b32bc

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

    def switch_frame(self):
        self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.1)
        self.browser.switch_to.frame(
            self.browser.find_element(
                "xpath", "/html/body/main/div[2]/div[2]/div/iframe"
            )
        )

    def send_comment(self, comment):
        element = self.browser.find_element(
            "xpath", "/html/body/main/article/form/div/textarea"
        )
        element.send_keys(comment)

    def click_xpath(self, element):
        element = self.browser.find_element("xpath", element)
        element.click()
        return element.text

    def next_post(self):
        self.browser.switch_to.default_content()
        _nxt = self.browser.find_element(
            "xpath", "/html/body/main/div[2]/div[1]/article/footer/div[2]/div[2]/a"
        )
        nxt = _nxt.text
        self.browser.get(_nxt.get_attribute("href"))
        return nxt

    def __call__(self):
        time.sleep(3)
        tmp = self.browser.find_element(
            "xpath", "/html/body/main/div[2]/div[1]/article/header/h1"
        ).text
        self.switch_frame()
        time.sleep(2)
        self.send_comment("test")
        time.sleep(0.1)
        self.click_xpath("/html/body/main/article/form/footer/button")
        time.sleep(2)
        self.browser.switch_to.default_content()
        return tmp, self.next_post()


def main():
    logger = zz.logging.Logger("UTTERANCE")
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
