from modules.movie_lightfm import (
    load_lightfm,
    get_lightfm_recommendations
)

model = load_lightfm()

print("\nMODEL INFORMATION")
print(model.get_model_info())

user_id = 196

recommendations = get_lightfm_recommendations(
    user_id=user_id,
    top_n=10
)

print("\nLIGHTFM RECOMMENDATIONS")

if recommendations.empty:
    print("No recommendations found.")
else:
    columns = [
        col for col in [
            "movie_id",
            "title",
            "score"
        ]
        if col in recommendations.columns
    ]

    print(recommendations[columns].to_string(index=False))