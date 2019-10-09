#1 Read in dataset
import pandas as pd
import numpy as np

#
apps_with_duplicates = pd.read_csv('datasets/apps.csv')

# Drop duplicates
apps = apps_with_duplicates.drop_duplicates()

# Print the total number of apps
print('Total number of apps in the dataset = ', apps.size)

# Have a look at a random sample of 5 entries
n = 5
print(apps.sample(n))

#2 Data cleaning

# Clean the special case columns
# Changing kB to MB by dividing by 1000
apps['Size'] = apps['Size'].apply(lambda x: str(float(x.replace('k', '')) / 1000) \
    if 'k' in x else x)
apps['Size'] = apps['Size'].replace('Varies with device', np.nan)

chars_to_remove = ['+', ',', 'M', '$']
cols_to_clean = ['Installs', 'Size', 'Price']
for col in cols_to_clean:
    # Remove the characters preventing us from converting to numeric
    for char in chars_to_remove:
        apps[col] = apps[col].str.replace(char, '')
    # Convert the column to numeric
    apps[col] = pd.to_numeric(apps[col])

#3 Android market breakdown
import plotly.graph_objs as go

# Print the total number of unique categories
num_categories = apps['Category'].nunique()
print('Number of categories = ', num_categories)

# Count the number of apps in each 'Category' and sort them for easier plotting
num_apps_in_category = apps['Category'].value_counts().sort_values(ascending=False)

fig = go.Figure(
    [go.Bar(
        x = num_apps_in_category.index, # index = category name
        y = num_apps_in_category.values, # value = count
        )],layout_title_text="The number of apps in each 'Category'"
    )


fig.show()

fig = go.Figure(
    [go.Pie(
        labels = num_apps_in_category.index, # index = category name
        values = num_apps_in_category.values, # value = count
        )]
    )


fig.show()

#4 Average rating of apps

avg_app_rating = apps['Rating'].mean()
print('Average app rating = ', avg_app_rating)

# Distribution of apps according to their ratings
data = [go.Histogram(
    x=apps['Rating'],
    xbins={'start': 1, 'size': 0.1, 'end': 5}
)]

# Vertical dashed line to indicate the average app rating
layout = {'shapes': [{
    'type': 'line',
    'x0': avg_app_rating,
    'y0': 0,
    'x1': avg_app_rating,
    'y1': 1000,
    'line': {'dash': 'dashdot'}
}]
}

fig = go.Figure({'data': data, 'layout': layout}, layout_title_text="Average rating of apps")
fig.show()


#5 Sizing and pricing strategy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings("ignore")

# Subset for categories with at least 250 apps
large_categories = apps.groupby('Category').filter(lambda x: len(x) >= 250).reset_index()

# Plot size vs. rating
sns.jointplot(x = large_categories['Size'], y = large_categories['Rating'], data = large_categories, kind = 'hex')

# Subset for paid apps only
paid_apps = apps[apps['Price'] > 0]

# Plot price vs. rating
sns.jointplot(x = paid_apps['Price'], y = paid_apps['Rating'], data = paid_apps)
#plt.show()

#6 How should you price your app?

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

# Select a few popular app categories
popular_app_cats = apps[apps.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY',
                                            'MEDICAL', 'TOOLS', 'FINANCE',
                                            'LIFESTYLE','BUSINESS'])]

# Examine the price trend for the subset of categories
ax = sns.stripplot(x='Price', y='Category', data=popular_app_cats,
                   jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories')

# Category, Name and Price of apps priced above $200
apps_above_200 = apps[apps['Price'] > 200][['Category', 'App', 'Price']]
print(apps_above_200)
#plt.show()

#7 Filter out "junk" apps

# Select apps priced below $100
apps_under_100 = popular_app_cats[popular_app_cats['Price'] < 100]

fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

# Examine price vs category with the authentic apps
ax = sns.stripplot(x='Price', y='Category', data=apps_under_100,
                   jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories after filtering for junk apps')
#plt.show()

#8 Number of installs for paid apps vs. free apps

trace0 = go.Box(
    # Data for paid apps
    y=apps[apps['Type'] == 'Paid']['Installs'],
    name = 'Paid'
)

trace1 = go.Box(
    # Data for free apps
    y=apps[apps['Type'] == 'Free']['Installs'],
    name = 'Free'
)

layout = go.Layout(
    title = "Number of downloads of paid apps vs. free apps",
    yaxis = dict(
        type = 'log',
        autorange = True
    )
)

# Add trace0 and trace1 to a list for plotting
data = [trace0, trace1]
#plotly.offline.iplot({'data': data, 'layout': layout})
fig = go.Figure({'data': data, 'layout': layout}, layout_title_text="Number of installs for paid apps vs. free apps")
fig.show()

#9 Sentiment analysis of user reviews
# Read in the user reviews
reviews_df = pd.read_csv('datasets/user_reviews.csv')

# Join and merge the two dataframe
merged_df = pd.merge(apps, reviews_df, on = 'App', how = "inner")

# Drop NA values from Sentiment and Translated_Review columns
merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)

# User review sentiment polarity for paid vs. free apps
ax = sns.boxplot(x = 'Type', y = 'Sentiment_Polarity', data = merged_df)
ax.set_title('Sentiment Polarity Distribution')
plt.show()
