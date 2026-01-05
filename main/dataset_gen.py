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

# -------------------------
# CONTEXT
# -------------------------
time_of_day = np.random.choice(
    ['morning', 'afternoon', 'night'], N, p=[0.3, 0.4, 0.3]
)

day_type = np.random.choice(
    ['weekday', 'weekend'], N, p=[0.7, 0.3]
)

# -------------------------
# CONTENT ATTRIBUTES
# -------------------------

# ProgramType depends on age, time of day, day type, gender, and household type
program_type = []
for age, time, gender, household, day in zip(
    user_age, time_of_day, user_gender, household_type, day_type
):
    if age == 'senior':
        if day == 'weekday':
            pt = np.random.choice(
                ['news', 'documentary', 'movie'], p=[0.6, 0.25, 0.15]
            )
        elif household == 'family':
            pt = np.random.choice(
                ['news', 'documentary', 'movie'], p=[0.55, 0.25, 0.2]
            )
        else:
            pt = np.random.choice(
                ['news', 'movie', 'documentary'], p=[0.5, 0.3, 0.2]
            )
    elif time == 'night':
        if day == 'weekend':
            pt = np.random.choice(
                ['movie', 'series', 'entertainment'], p=[0.45, 0.35, 0.2]
            )
        elif household == 'family':
            pt = np.random.choice(
                ['series', 'entertainment', 'movie'], p=[0.45, 0.35, 0.2]
            )
        elif gender == 'female':
            pt = np.random.choice(
                ['series', 'movie', 'entertainment'], p=[0.45, 0.4, 0.15]
            )
        else:
            pt = np.random.choice(
                ['movie', 'series', 'entertainment'], p=[0.55, 0.3, 0.15]
            )
    else:
        if day == 'weekend':
            pt = np.random.choice(
                ['entertainment', 'series', 'news'], p=[0.5, 0.35, 0.15]
            )
        elif household == 'family':
            pt = np.random.choice(
                ['entertainment', 'series', 'news'], p=[0.5, 0.35, 0.15]
            )
        elif gender == 'female':
            pt = np.random.choice(
                ['series', 'entertainment', 'news'], p=[0.4, 0.35, 0.25]
            )
        else:
            pt = np.random.choice(
                ['entertainment', 'news', 'series'], p=[0.4, 0.35, 0.25]
            )
    program_type.append(pt)

# ProgramGenre depends on ProgramType, day type, gender, and household type
program_genre = []
for pt, gender, household, day in zip(
    program_type, user_gender, household_type, day_type
):
    if pt == 'news':
        genre = 'news'
    elif pt == 'documentary':
        genre = 'documentary'
    elif pt == 'movie':
        if household == 'family':
            genre = np.random.choice(
                ['comedy', 'drama', 'romance', 'horror'], p=[0.4, 0.35, 0.2, 0.05]
            )
        elif gender == 'female':
            genre = np.random.choice(
                ['drama', 'romance', 'comedy', 'horror'], p=[0.45, 0.3, 0.2, 0.05]
            )
        else:
            genre = np.random.choice(
                ['drama', 'horror', 'comedy', 'romance'], p=[0.35, 0.25, 0.25, 0.15]
            )
    elif pt == 'series':
        if household == 'family':
            genre = np.random.choice(
                ['comedy', 'drama', 'horror'], p=[0.5, 0.4, 0.1]
            )
        elif gender == 'female':
            genre = np.random.choice(
                ['drama', 'comedy', 'horror'], p=[0.45, 0.45, 0.1]
            )
        else:
            genre = np.random.choice(
                ['drama', 'comedy', 'horror'], p=[0.35, 0.4, 0.25]
            )
    else:  # entertainment
        if day == 'weekend':
            genre = np.random.choice(
                ['entertainment', 'comedy'], p=[0.7, 0.3]
            )
        else:
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
    'TimeOfDay': time_of_day,
    'DayType': day_type,
    'ProgramType': program_type,
    'ProgramGenre': program_genre,
    'ProgramDuration': program_duration
})

df_profile.to_csv("main/consumers_profile.csv", index=False)

print("✅ ConsumersProfile dataset generated successfully")
print(df_profile.head())
