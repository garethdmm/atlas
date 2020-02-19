from datetime import datetime

"""
We have essentially 24 months of data. We break it up as follows:
60% Training ~ 14.4 months, January 2016 - March 15th, 2016
20% Validation ~ 4.8 months, March 16th, 2016 - September 1st, 2016
20% Testing ~ 4.8 months, September 1st, 2016 - January 1st, 2016

We will likely have to modify these for different featuresets that might not have
complete data in these periods.
"""

TRAIN_START = datetime(2015, 1, 1)
TRAIN_END = datetime(2016, 3, 15)

TEST_START = datetime(2016, 3, 16)
TEST_END = datetime(2016, 9, 1)

# Some featuresets have their first valid index much later than the default for various
# reasons. For these we give them different train/test windows since the default would
# give them too much data in the test window.

INTEREXCHANGE_TRAIN_START = datetime(2015, 11, 11)
INTEREXCHANGE_TRAIN_END = datetime(2016, 5, 1)

INTEREXCHANGE_TEST_START = datetime(2016, 5, 2)
INTEREXCHANGE_TEST_END = datetime(2016, 9, 1)

# We have 2015-11-11 to 2016-9 for Gemini, 10.67 months. Split it 75/25 and we get:
GEMINI_TRAIN_START = datetime(2015, 11, 15)
GEMINI_TRAIN_END = datetime(2016, 6, 1)

GEMINI_TEST_START = datetime(2016, 6, 2)
GEMINI_TEST_END = datetime(2016, 9, 1)

# We have 2015-4 to 2016-9 for OKCoin, 18 months. Split it 75/25 we get this:
OKCOIN_TRAIN_START = datetime(2015, 4, 4)
OKCOIN_TRAIN_END = datetime(2016, 5, 1)

OKCOIN_TEST_START = datetime(2016, 5, 15)
OKCOIN_TEST_END = datetime(2016, 9, 1)

LEARNING_RATE=0.01
EPOCHS=20
DISPLAY_STEP = 10
