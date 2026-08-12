from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
 
 
class UserProfile(models.Model):
    """
    One health/dietary profile per user - everything the recommendation
    engine needs to personalize meal plans lives here.
    """
 
    class Gender(models.TextChoices):
        MALE = "male", "Male"
        FEMALE = "female", "Female"
 
    class Goal(models.TextChoices):
        LOSE_WEIGHT = "lose_weight", "Lose weight"
        MAINTAIN_WEIGHT = "maintain_weight", "Maintain weight"
        GAIN_WEIGHT = "gain_weight", "Gain weight"
        GAIN_MUSCLE = "gain_muscle", "Gain muscle"
 
    class ActivityLevel(models.TextChoices):
        SEDENTARY = "sedentary", "Sedentary (little or no exercise)"
        LIGHT = "light", "Lightly active (1-3 days/week)"
        MODERATE = "moderate", "Moderately active (3-5 days/week)"
        ACTIVE = "active", "Active (6-7 days/week)"
        VERY_ACTIVE = "very_active", "Very active (athlete / physical job)"
 
    class MealType(models.TextChoices):
        STANDARD = "standard", "Standard"
        VEGETARIAN = "vegetarian", "Vegetarian"
        VEGAN = "vegan", "Vegan"
        KETO = "keto", "Keto"
        HALAL = "halal", "Halal"
        OTHER = "other", "Other"
 
    # settings.AUTH_USER_MODEL (not User directly) so this still works if
    # you ever swap in a custom user model later.
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )
 
    age = models.PositiveIntegerField(validators=[MinValueValidator(1), MaxValueValidator(120)])
    height = models.FloatField(
        help_text="Height in centimeters",
        validators=[MinValueValidator(50), MaxValueValidator(250)],
    )
    weight = models.FloatField(
        help_text="Weight in kilograms",
        validators=[MinValueValidator(20), MaxValueValidator(400)],
    )
    gender = models.CharField(max_length=10, choices=Gender.choices)
    pregnant = models.BooleanField(default=False)
 
    # Each stored as a JSON list of strings, e.g. ["diabetes", "hypertension"]
    conditions = models.JSONField(default=list, blank=True)
    allergies = models.JSONField(default=list, blank=True)
    preferences = models.JSONField(default=list, blank=True)
    taste_text = models.TextField(blank=True)
 
    goal = models.CharField(max_length=20, choices=Goal.choices)
    activity_level = models.CharField(max_length=20, choices=ActivityLevel.choices)
    meal_type = models.CharField(max_length=20, choices=MealType.choices)
 
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
 
    def __str__(self):
        return f"{self.user.username}'s profile"
 