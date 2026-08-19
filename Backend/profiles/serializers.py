from rest_framework import serializers

from .models import UserProfile


class UserProfileSerializer(serializers.ModelSerializer):
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

        gender = attrs.get("gender", getattr(self.instance, "gender", None))
        pregnant = attrs.get("pregnant", getattr(self.instance, "pregnant", False))

        if pregnant and gender != UserProfile.Gender.FEMALE:
            raise serializers.ValidationError(
                {"pregnant": "Can only be true when gender is female."}
            )
        return attrs