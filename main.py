import logging
from enum import Enum

LOGGING_MODE = logging.CRITICAL
SELECTED_MENU = 0


class MENU(Enum):
    EXIT = -1
    HALT = 0
    TRAIN = 1
    TEST = 2
    PREDICT = 3
    LIVE = 4
    BACKTEST = 5

    # If not found, return HALT
    @staticmethod
    def _missing_(value):
        return MENU.HALT


def menu():
    global SELECTED_MENU
    if SELECTED_MENU == MENU.TRAIN:
        from deep_learning import train
        train()
    elif SELECTED_MENU == MENU.TEST:
        from deep_learning import test
        test()
    elif SELECTED_MENU == MENU.PREDICT:
        from deep_learning import predict
        predict()
    elif SELECTED_MENU == MENU.LIVE:
        from trading import live
        live()
    elif SELECTED_MENU == MENU.BACKTEST:
        from trading import backtest
        backtest()
    else:
        logging.error("Invalid selection: %s", SELECTED_MENU)


def main():
    global SELECTED_MENU
    logging.basicConfig(level=LOGGING_MODE, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s')
    logging.info('Logging mode: %s', logging.getLevelName(LOGGING_MODE))
    while SELECTED_MENU != MENU.EXIT:
        print("Input a number to select a menu: ")
        print("---------------------------------")
        print(" 1. | Train")
        print(" 2. | Test")
        print(" 3. | Predict")
        print(" 4. | Live Prediction")
        print(" 5. | Backtest")
        print("-1. | Exit")
        print("---------------------------------")
        console_input = input("Enter: ")
        try:
            SELECTED_MENU = MENU(int(console_input))
        except ValueError:
            SELECTED_MENU = MENU.HALT
            try:
                exec(console_input)
            except Exception as e:
                logging.error(e)
        if not SELECTED_MENU == MENU.HALT:
            print("Selection: %s" % SELECTED_MENU)
            if SELECTED_MENU == MENU.EXIT:
                break
            menu()


if __name__ == '__main__':
    SELECTED_MENU = MENU.HALT
    main()
