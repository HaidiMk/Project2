from rest_framework import serializers

from .models import UserProfile


class UserProfileSerializer(serializers.ModelSerializer):
    """
    Handles both reading and writing UserProfile data.

    Numeric-range checks (age/height/weight limits) live on the model
    fields themselves via validators=[...] in models.py - ModelSerializer
    picks those up automatically, so they aren't repeated here.
    """

    class Meta:
        model = UserProfile
        fields = [
            "age", "height", "weight", "gender", "pregnant",
            "conditions", "allergies", "preferences", "taste_text",
            "goal", "activity_level", "meal_type",
            "updated_at",
        ]
        read_only_fields = ["updated_at"]

    def validate(self, attrs):
        """
        Object-level check: 'pregnant' only makes sense when gender=female.
        Falls back to the existing instance's values on a partial update
        (PATCH), where the client might send only one of the two fields.
        """
        gender = attrs.get("gender", getattr(self.instance, "gender", None))
        pregnant = attrs.get("pregnant", getattr(self.instance, "pregnant", False))

        if pregnant and gender != UserProfile.Gender.FEMALE:
            raise serializers.ValidationError(
                {"pregnant": "Can only be true when gender is female."}
            )
        return attrs