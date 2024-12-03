import nltk
import pandas as pd


def load_data(input_file):
    df = pd.read_csv(input_file)
    return df

def create_key(df):
    df = df.copy()
    df["key"] = df["raw_text"]
    df["key"] = df["key"].str.strip()
    df["key"] = df["key"].str.lower()
    df["key"] = df["key"].str.replace("-","") #using the translate one below is also sufficient
    df["key"] = df["key"].str.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]_^{|}~/"))
    df["key"] = df["key"].str.split()

    #Root of every word
    stemmer = nltk.PorterStemmer()
    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])

    # Uniqueness and alphabetical order check
    df["key"] = df["key"].apply(lambda x: sorted(set(x)))

    # we turn them back into a string, now we can group same items equally
    df["key"] = df["key"].str.join(" ")

    return df


def generate_cleaned_column(df):
    df = df.copy()
    # df_ori = df.copy()

    # df = df.sort_values(by=["key", "raw_text"], ascending=[True, True]) # the test asks for another order thus this line is commented now

    # print(df)
    # df["i2u"] = ""

    # for i in range(len(df["key"])-1):
    #     if i == 0:
    #         df["i2u"].iloc[i] = df["key"].iloc[i]
    #     elif df["key"].iloc[i] != df["key"].iloc[i+1]:
    #         df["i2u"].iloc[i+1] = df["key"].iloc[i+1]

    # pick the first row of each "key" group
    keys = df.drop_duplicates(subset="key",keep="first")
    
    # tuple baes dictionary with key as key and val as text
    key_dict = dict(zip(keys["key"], keys["raw_text"]))

    df["cleaned"] = df["key"].map(key_dict)

    return df


def save_data(df, output_file):
    df = df.copy()
    df = df[["cleaned"]]
    # df = df.rename(columns={"cleaned": "text"}) same thing, test asks for something different from the workshop during class
    df = df.rename(columns={"cleaned": "cleaned_text"})
    df.to_csv(output_file, index = False)

def main(input_file, output_file):
    df = load_data(input_file)
    df = create_key(df)
    df = generate_cleaned_column(df)
    print(df)
    df.to_csv("./files/test.csv", index=False)
    save_data(df, output_file)

if __name__ == "__main__":
    main("./files/input.txt", "./files/output.txt")