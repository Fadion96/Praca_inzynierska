from django.db import models


# Create your models here.


def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    return 'uploads/{0}/{1}'.format(instance.session_id, filename)


class SessionManager(models.Manager):
    def create_session(self):
        session = self.create()
        return session


class Session(models.Model):
    session_id = models.AutoField(primary_key=True)
    objects = SessionManager()


class ProcessingFunction(models.Model):
    name = models.CharField(max_length=30)
    function = models.CharField(max_length=30)
    number_of_images = models.PositiveSmallIntegerField(default=1)
    number_of_parameters = models.PositiveSmallIntegerField(default=0)
    params = models.TextField(blank=True, null=True)
    type = models.CharField(max_length=10)


class UserFunctionManager(models.Manager):
    def create_user_function(self, session_id, name, file):
        user_function = self.create(session_id=session_id, name=name, file=file)
        return user_function


class UserFunction(models.Model):
    session_id = models.IntegerField()
    name = models.CharField(max_length=30)
    file = models.FileField(upload_to=user_directory_path)
    objects = UserFunctionManager()
