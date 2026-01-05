import pandas as pd
import numpy as np

# -------------------------
# CONFIGURATION
# -------------------------
N = 10000
np.random.seed(42)

# -------------------------
# USER PROFILE
# -------------------------
user_age = np.random.choice(
    ['young', 'adult', 'senior'], N, p=[0.4, 0.4, 0.2]
)

user_gender = np.random.choice(
    ['male', 'female'], N, p=[0.5, 0.5]
)

household_type = np.random.choice(
    ['single', 'couple', 'family'], N, p=[0.3, 0.4, 0.3]
)

tech_level = np.random.choice(
    ['low', 'medium', 'high'], N, p=[0.3, 0.5, 0.2]
)

# -------------------------
# CONTEXT
# -------------------------
time_of_day = np.random.choice(
    ['morning', 'afternoon', 'night'], N, p=[0.3, 0.4, 0.3]
)

day_type = np.random.choice(
    ['weekday', 'weekend'], N, p=[0.7, 0.3]
)

special_period = np.random.choice(
    ['none', 'christmas', 'summer'], N, p=[0.75, 0.15, 0.10]
)

# -------------------------
# CONTENT ATTRIBUTES
# -------------------------

# ProgramType depends on age and time of day
program_type = []
for age, time in zip(user_age, time_of_day):
    if age == 'senior':
        pt = np.random.choice(
            ['news', 'movie', 'documentary'], p=[0.5, 0.3, 0.2]
        )
    elif time == 'night':
        pt = np.random.choice(
            ['movie', 'series', 'entertainment'], p=[0.5, 0.3, 0.2]
        )
    else:
        pt = np.random.choice(
            ['entertainment', 'series', 'news'], p=[0.4, 0.3, 0.3]
        )
    program_type.append(pt)

# ProgramGenre depends on ProgramType
program_genre = []
for pt in program_type:
    if pt == 'news':
        genre = 'news'
    elif pt == 'documentary':
        genre = 'documentary'
    elif pt == 'movie':
        genre = np.random.choice(
            ['drama', 'horror', 'comedy', 'romance'], p=[0.4, 0.2, 0.2, 0.2]
        )
    elif pt == 'series':
        genre = np.random.choice(
            ['drama', 'comedy', 'horror'], p=[0.4, 0.4, 0.2]
        )
    else:  # entertainment
        genre = 'entertainment'
    program_genre.append(genre)

# ProgramDuration depends on ProgramType
program_duration = []
for pt in program_type:
    if pt == 'movie':
        duration = 'long'
    elif pt == 'series':
        duration = 'medium'
    elif pt == 'news':
        duration = 'short'
    else:
        duration = np.random.choice(
            ['short', 'medium'], p=[0.6, 0.4]
        )
    program_duration.append(duration)

# -------------------------
# FINAL DATASET
# -------------------------
df_profile = pd.DataFrame({
    'UserAge': user_age,
    'UserGender': user_gender,
    'HouseholdType': household_type,
    'TechLevel': tech_level,
    'TimeOfDay': time_of_day,
    'DayType': day_type,
    'SpecialPeriod': special_period,
    'ProgramType': program_type,
    'ProgramGenre': program_genre,
    'ProgramDuration': program_duration
})

df_profile.to_csv("consumers_profile.csv", index=False)

print("✅ ConsumersProfile dataset generated successfully")
print(df_profile.head())
