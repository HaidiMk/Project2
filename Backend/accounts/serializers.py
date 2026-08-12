from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers


class UserSerializer(serializers.ModelSerializer):
    """
    Read-only serializer used to shape the User object in API responses.
    This is the DRF equivalent of a Laravel API Resource (e.g. UserResource) -
    it controls exactly what user fields are exposed to the client.
    """

    class Meta:
        model = User
        fields = ("id", "username", "email")


class RegisterSerializer(serializers.ModelSerializer):
    """
    Handles both validation AND creation of a new user (Sign Up).

    Unlike Laravel, where validation (FormRequest) and the actual
    User::create() call usually live in different places, DRF's
    ModelSerializer intentionally bundles both in one class:
      - validate_<field>() / validate()  -> validation rules
      - create()                         -> what happens once validation passes
    """

    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],
        style={"input_type": "password"},
    )
    password2 = serializers.CharField(
        write_only=True,
        required=True,
        label="Confirm password",
        style={"input_type": "password"},
    )

    class Meta:
        model = User
        fields = ("username", "email", "password", "password2")
        extra_kwargs = {
            "email": {"required": True},
        }

    def validate_email(self, value):
        """Field-level validation: DRF runs this automatically for 'email'."""
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value

    def validate(self, attrs):
        """Object-level validation: runs after every field passes on its own."""
        if attrs["password"] != attrs["password2"]:
            raise serializers.ValidationError({"password2": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        """
        Called internally by serializer.save().

        We MUST use create_user(), not User.objects.create(), because
        create_user() hashes the password via set_password() for us.
        Using create() directly would store the raw password as plain text.
        """
        validated_data.pop("password2")
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"],
        )
        return user


class LoginSerializer(serializers.Serializer):
    """
    يقبل اسم المستخدم أو البريد الإلكتروني لتسجيل الدخول.
    """
    # جعلنا الحقلين (اختياريين) من ناحية جانغو لنسمح بمرور الطلب إذا كان أحدهما مفقوداً
    username = serializers.CharField(required=False)
    email = serializers.EmailField(required=False)
    password = serializers.CharField(write_only=True, style={"input_type": "password"})

    def validate(self, attrs):
        username = attrs.get("username")
        email = attrs.get("email")
        password = attrs.get("password")

        if not username and not email:
            raise serializers.ValidationError("the username or email is required ")
        
        if email:
            try:
                user_obj = User.objects.get(email__iexact=email)
                login_id = user_obj.username 
            except User.DoesNotExist:
                login_id = "invalid_user" 
        else:
            login_id = username

        user = authenticate(
            username=login_id,
            password=password,
        )

        if not user:
            raise serializers.ValidationError(
                code="authorization"
            )

        attrs["user"] = user
        return attrs