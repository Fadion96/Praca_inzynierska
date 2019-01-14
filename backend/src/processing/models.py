from django.db import models


# Create your models here.


def user_directory_path(instance, filename):
    """
    Funkcja tworząca ścieżki do programów dodawanych przez użytkownika.
    :param instance: tworzona instancja
    :param filename: nazwa pliku
    :return: ścieżka pod którą zostanie zapisany plik
    """
    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    return 'uploads/{0}/{1}'.format(instance.session_id, filename)


class SessionManager(models.Manager):
    """
    Klasa będąca mananagerem dla modelu Session.
    """
    def create_session(self):
        """
        Funkcja, która tworzy instancje modelu Session.
        :return: stworzona instancja modelu Session.
        """
        session = self.create()
        return session


class Session(models.Model):
    """
    Model opisujący sesje programu.
    """
    session_id = models.AutoField(primary_key=True)
    objects = SessionManager()


class ProcessingFunction(models.Model):
    """
    Model określający zaimplementowane algorytmy przetwarzania obrazu w programie przechowywane w bazie danych.
    """
    name = models.CharField(max_length=30)
    function = models.CharField(max_length=30)
    number_of_images = models.PositiveSmallIntegerField(default=1)
    number_of_parameters = models.PositiveSmallIntegerField(default=0)
    params = models.TextField(blank=True, null=True)
    type = models.CharField(max_length=10)


class UserFunctionManager(models.Manager):
    """
    Klasa będąca mananagerem dla modelu UserFunction.
    """
    def create_user_function(self, session_id, name, file):
        """
        Funkcja, która tworzy instancje modelu UserFunction.
        :return: stworzona instancja modelu UserFunction.
        """
        user_function = self.create(session_id=session_id, name=name, file=file)
        return user_function


class UserFunction(models.Model):
    """
    Model określający dodawane przez użytkowników algorytmy przetwarzania obrazu.
    """
    session_id = models.IntegerField()
    name = models.CharField(max_length=30)
    file = models.FileField(upload_to=user_directory_path)
    objects = UserFunctionManager()
