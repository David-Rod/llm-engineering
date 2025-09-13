# %%
import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt
import json
from collections import Counter
import pickle
import random

import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from openai import OpenAI
from anthropic import Anthropic


# %%
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')

os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY', 'your-wandb-key')

# %%
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

# %%
from items import Item
from loaders import ItemLoader
from testing import Tester

# %%
openai = OpenAI()
claude = Anthropic()

# %%
%matplotlib inline

# %%
# Specify and load huggingface dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_Appliances", split="full", trust_remote_code=True)

# %%
print(f"Number of Appliances: {len(dataset):,}")
datapoint = dataset[4]
print(f"Title: {datapoint['title']}\n")
print(f"Description: {datapoint['description']}\n")
print(f"Features: {datapoint['features']}\n")
print(f"Details: {datapoint['details']}\n")
print(f"Price: {datapoint['price']}\n")

# %%
##################### GET CATEGORIES #####################
def get_categories_data(dataset):
    category_counter = Counter()
    categories_list = []
    for datapoint in dataset:
        try:
            category = datapoint['details']
            category_as_json = json.loads(category)
            keys = category_as_json.keys()
            for key in keys:
                if key == "Best Sellers Rank":
                    item_category = list(category_as_json[key].keys())
                    if item_category and isinstance(item_category, list):
                        for cat in item_category:
                            category_counter[cat] += 1
                        categories_list.append(item_category)
        except KeyError:
            pass

    return category_counter, categories_list


# %%
category_counter, categories_list_per_item= get_categories_data(dataset)

# %%
category_counter.elements

# %%
display_counter = category_counter.most_common(20)
plt.figure(figsize=(20, 15))
plt.title(f"Category Distribution: Total items {sum(category_counter.values())}\n")
plt.xlabel('Category Count')
plt.ylabel('Category names')
plt.barh([item[0] for item in display_counter], [item[1] for item in display_counter], color="purple")
plt.show()

# %%
no_empty_categories = [item for item in categories_list_per_item if item]

print(f"There are {len(no_empty_categories):,} with categories which is {len(no_empty_categories)/len(dataset)*100:,.1f}%")

# %%
##################### GET PRICES #####################
prices = 0
for datapoint in dataset:
    try:
        price = float(datapoint["price"])
        if price > 0:
            prices += 1
    except ValueError as e:
        pass

print(f"There are {prices:,} with prices which is {prices/len(dataset)*100:,.1f}%")

# %%
def get_prices(dataset):
    prices = []
    len_prices = []
    for datapoint in dataset:
        try:
            price = float(datapoint["price"])
            if price > 0:
                prices.append(price)
                contents = datapoint["title"] + str(datapoint["description"]) + str(datapoint["features"]) + str(datapoint["details"])
                len_prices.append(len(contents))
        except ValueError as e:
            pass
    return prices, len_prices

# %%
prices, len_prices = get_prices(dataset)
print(prices[:10])
print(len_prices[:10])

# %%
plt.figure(figsize=(15, 6))
plt.title(f"Lengths: Avg {sum(len_prices)/len(len_prices):,.0f} and highest {max(len_prices):,}\n")
plt.xlabel('Length (chars)')
plt.ylabel('Count')
plt.hist(len_prices, rwidth=0.7, color="lightblue", bins=range(0, 6000, 100))
plt.show()

# %%
plt.figure(figsize=(15, 6))
plt.title(f"Prices: Avg {sum(prices)/len(prices):,.2f} and highest {max(prices):,}\n")
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.hist(prices, rwidth=0.7, color="orange", bins=range(0, 1000, 10))
plt.show()

# %%
tokens = [item.token_count for item in items]
plt.figure(figsize=(15, 6))
plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}\n")
plt.xlabel('Length (tokens)')
plt.ylabel('Count')
plt.hist(tokens, rwidth=0.7, color="green", bins=range(0, 300, 10))
plt.show()

# %%
prices = [item.price for item in items]
plt.figure(figsize=(15, 6))
plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.hist(prices, rwidth=0.7, color="purple", bins=range(0, 300, 10))
plt.show()

# %%
dataset_names = [
    # "Automotive",
    # "Electronics",
    # "Office_Products",
    # "Tools_and_Home_Improvement",
    # "Cell_Phones_and_Accessories",
    # "Toys_and_Games",
    "Appliances",
    # "Musical_Instruments",
]

# %%
items = []
for dataset_name in dataset_names:
    loader = ItemLoader(dataset_name)
    items.extend(loader.load())

# %%
tokens = [item.token_count for item in items]
plt.figure(figsize=(15, 6))
plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}\n")
plt.xlabel('Length (tokens)')
plt.ylabel('Count')
plt.hist(tokens, rwidth=0.7, color="skyblue", bins=range(0, 300, 10))
plt.show()

# %%
prices = [item.price for item in items]
plt.figure(figsize=(15, 6))
plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
plt.show()

# %%
sample = items

sizes = [len(item.prompt) for item in sample]
prices = [item.price for item in sample]

# Create the scatter plot
plt.figure(figsize=(15, 8))
plt.scatter(sizes, prices, s=0.2, color="red")

# Add labels and title
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Is there a simple correlation?')

# Display the plot
plt.show()

# %%
def report(item):
    prompt = item.prompt
    tokens = Item.tokenizer.encode(item.prompt)
    print(prompt)
    print(tokens[-10:])
    print(Item.tokenizer.batch_decode(tokens[-10:]))

# %%
report(sample[0])

# %%
random.seed(42)
random.shuffle(sample)
train = sample[:26_000]
test = sample[26_000:27_000]
print(f"Divided into a training set of {len(train):,} items and test set of {len(test):,} items")

# %%
# Observe how tokens are handles when 3 digits ($100 or more)
max_item = [item for item in items if item.price > 99]
report(max_item[0])

# %%
train_prompts = [item.prompt for item in train]
train_prices = [item.price for item in train]
test_prompts = [item.test_prompt() for item in test]
test_prices = [item.price for item in test]

# %%
train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# %%
DATASET_NAME = "david-rod/lite-data"
dataset.push_to_hub(DATASET_NAME, private=True)

# %%

with open('train_lite.pkl', 'wb') as file:
    pickle.dump(train, file)

with open('test_lite.pkl', 'wb') as file:
    pickle.dump(test, file)

# %%
print(train[0].prompt)

# %%
print(train[0].price)

# %%
print(train[0].test_prompt())

# %%
class Tester:

    def __init__(self, predictor, title=None, data=test, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []

    def color_for(self, error, truth):
        if error/truth < 0.1:
            return "green"
        elif error/truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        error = abs(guess - truth)
        log_error = math.log(truth+1) - math.log(guess+1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40]+"..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} Error_Perc={error/truth:.2%}% SLE: {sle:,.2f} Item: {title}{RESET}")

    def chart(self, title):
        max_error = max(self.errors)
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color=="green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function):
        cls(function).run()

# %%
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}

# %%
def random_pricer(item):
    return random.randrange(1,1000)

# %%
random.seed(42)
print(random.randrange(1,1000))
print(random.randrange(1,1000))
print(random.randrange(1,1000))

# %%
random.seed(42)

Tester.test(random_pricer)

# %%
training_prices = [item.price for item in train]
training_average = sum(training_prices) / len(training_prices)

def constant_pricer(item):
    return training_average

# %%
Tester.test(constant_pricer)

# %%
train[0].details

# %%
# Create a new "features" field on items, and populate it with json parsed from the details dict

for item in train:
    item.features = json.loads(item.details)
for item in test:
    item.features = json.loads(item.details)

# Look at one
train[0].features.keys()

# %%
feature_count = Counter()
for item in train:
    for f in item.features.keys():
        feature_count[f]+=1

feature_count.most_common(40)

# %%
def get_brands(item):
    return item.features.get("Brand", "Unknown")

# %%
def get_weight(item):
    weight_str = item.features.get('Item Weight')
    if weight_str:
        parts = weight_str.split(' ')
        amount = float(parts[0])
        unit = parts[1].lower()
        if unit=="pounds":
            return amount
        elif unit=="ounces":
            return amount / 16
        elif unit=="grams":
            return amount / 453.592
        elif unit=="milligrams":
            return amount / 453592
        elif unit=="kilograms":
            return amount / 0.453592
        elif unit=="hundredths" and parts[2].lower()=="pounds":
            return amount / 100
        else:
            print(weight_str)
    return None

weights = [get_weight(t) for t in train]
weights = [w for w in weights if w]
average_weight = sum(weights)/len(weights)

def get_weight_with_default(item):
    weight = get_weight(item)
    return weight or average_weight

# %%
def get_product_dimensions(item):
    dims_str = item.features.get("Product Dimensions", "Unknown")
    if dims_str == "Unknown" or not dims_str:
        return None

    try:
        # Extract numbers from string like "10 x 10 x 2.5 inches"
        import re
        numbers = re.findall(r'\d+\.?\d*', dims_str)
        if len(numbers) >= 3:
            length, width, height = float(numbers[0]), float(numbers[1]), float(numbers[2])
            volume = length * width * height
            return volume
        elif len(numbers) == 2:
            # For 2D items, assume height = 1
            length, width = float(numbers[0]), float(numbers[1])
            return length * width
    except (ValueError, IndexError):
        pass

    return None

volumes = [get_product_dimensions(item) for item in train]
volumes = [v for v in volumes if v is not None]
average_volume = sum(volumes) / len(volumes) if volumes else 1.0

def get_product_dimensions_with_default(item):
    volume = get_product_dimensions(item)
    return volume if volume is not None else average_volume

# %%
def get_text_length(item):
    return len(item.test_prompt())

# %%
def get_category(item):
    if item.features.get("Best Sellers Rank"):
        categories = list(item.features.get("Best Sellers Rank").keys())
        for category in categories:
            if "Appliances" in category:
                return category

# %%
brands = Counter()
for t in train:
    brand = t.features.get("Brand")
    if brand:
        brands[brand]+=1

brands.most_common(40)

# %%
TOP_APPLIANCE_BRANDS = [brand for brand, _ in brands.most_common(50)]
def is_top_appliance_brand(item):
    brand = item.features.get("Brand")
    # Handle both None and string "None" as no brand
    if not brand or brand == "None":
        return False
    return brand in TOP_APPLIANCE_BRANDS

# %%
# Items in dict indicate features considered for price impact
def get_features(item):
    return {
        "category": 1 if get_category(item) else 0,
        "weight": get_weight_with_default(item),
        "text_length": get_text_length(item),
        "is_top_appliance_brand": 1 if is_top_appliance_brand(item) else 0
    }

# %%
def list_to_dataframe(items):
    features = [get_features(item) for item in items]
    df = pd.DataFrame(features)
    df['price'] = [item.price for item in items]
    return df

train_df = list_to_dataframe(train)
test_df = list_to_dataframe(test[:250])

# %%
train_df.head()

# %%
np.random.seed(42)

# Separate features and target
feature_columns = ['category', 'weight', 'text_length', 'is_top_appliance_brand']

X_train = train_df[feature_columns]
y_train = train_df['price']
X_test = test_df[feature_columns]
y_test = test_df['price']

# Train a Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

for feature, coef in zip(feature_columns, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")

# Predict the test set and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# %%
def linear_regression_pricer(item):
    features = get_features(item)
    features_df = pd.DataFrame([features])
    return model.predict(features_df)[0]

# %%
Tester.test(linear_regression_pricer)

# %%
#### BAG OF WORDS ####

# For the next few models, we prepare our documents and prices
# Note that we use the test prompt for the documents, otherwise we'll reveal the answer!!

prices = np.array([float(item.price) for item in train])
documents = [item.test_prompt() for item in train]

# %%
np.random.seed(42)
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)
regressor = LinearRegression()
regressor.fit(X, prices)

# %%
def bow_lr_pricer(item):
    x = vectorizer.transform([item.test_prompt()])
    return max(regressor.predict(x)[0], 0)

# %%
Tester.test(bow_lr_pricer)

# %%
with open('train_lite.pkl', 'rb') as file:
    train = pickle.load(file)

with open('test_lite.pkl', 'rb') as file:
    test = pickle.load(file)

# %%
def messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"}
    ]

# %%
messages_for(test[0])

# %%
import re


def get_price(string_value):
    s = string_value.replace('$','').replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0

# %%
get_price("The price is roughly $99.99 because blah blah")

# %%
def gpt_4o_mini(item):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_for(item),
        seed=42,
        max_tokens=5
    )
    reply = response.choices[0].message.content
    return get_price(reply)

# %%
test[0].price


# %%
Tester.test(gpt_4o_mini, test)

# %%
def gpt_4o_frontier(item):
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages_for(item),
        seed=42,
        max_tokens=5
    )
    reply = response.choices[0].message.content
    return get_price(reply)

# %%
Tester.test(gpt_4o_frontier, test)

# %%
def claude_3_point_5_sonnet(item):
    messages = messages_for(item)
    system_message = messages[0]['content']
    messages = messages[1:]
    response = claude.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=5,
        system=system_message,
        messages=messages
    )
    reply = response.content[0].text
    return get_price(reply)

# %%
Tester.test(claude_3_point_5_sonnet, test)

# %%
fine_tune_train = train[:200]
fine_tune_validation = train[200:250]

# %%
def messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": f"Price is ${item.price:.2f}"}
    ]

# %%
messages_for(train[0])

# %%
def make_jsonl(items):
    result = ""
    for item in items:
        messages = messages_for(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str +'}\n'
    return result.strip()

# %%
print(make_jsonl(train[:3]))

# %%
def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)

# %%
write_jsonl(fine_tune_train, "fine_tune_train_lite.jsonl")

# %%
write_jsonl(fine_tune_validation, "fine_tune_validation_lite.jsonl")

# %%
with open("fine_tune_train_lite.jsonl", "rb") as f:
    train_file = openai.files.create(file=f, purpose="fine-tune")

with open("fine_tune_validation_lite.jsonl", "rb") as f:
    validation_file = openai.files.create(file=f, purpose="fine-tune")


# %%
train_file

# %%
validation_file

# %%
wandb_integration = {"type": "wandb", "wandb": {"project": "gpt-pricer"}}

# %%
# import wandb
# # Initialize W&B
# wandb.init(project="gpt-pricer", name="fine-tune-experiment")

# # Log your training parameters
# wandb.config.update({
#     "model": "gpt-4o-mini-2024-07-18",
#     "training_examples": len(fine_tune_train),
#     "validation_examples": len(fine_tune_validation),
#     "epochs": 1
# })

# %%
train_file.id

# %%
openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model="gpt-4o-mini-2024-07-18",
    seed=42,
    hyperparameters={"n_epochs": 1},
    integrations = [wandb_integration],
    suffix="pricer"
)

# %%
openai.fine_tuning.jobs.list(limit=1)

# %%
job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id

# %%
job_id

# %%
openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data

# %%
job = openai.fine_tuning.jobs.retrieve(job_id)
print(f"Status: {job.status}")
print(f"Model: {job.fine_tuned_model}")

# %%
fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model

# %%
fine_tuned_model_name

# %%
def gpt_fine_tuned(item):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name,
        messages=messages_for(item),
        seed=42,
        max_tokens=7
    )
    reply = response.choices[0].message.content
    return get_price(reply)

# %%
print(test[0].price)
print(gpt_fine_tuned(test[0]))

# %%
print(test[0].test_prompt())

# %%
Tester.test(gpt_fine_tuned, test)


