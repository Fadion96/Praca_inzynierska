virtualenv env
call env\Scripts\activate
call env\Scripts\pip.exe install -r requirements.txt
cd backend\src
python manage.py makemigrations
python manage.py migrate
python manage.py loaddata db.json
cd ..
cd ..