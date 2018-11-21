from database.database import Database


def run():
    db = Database('/home/spolu/test.sqlite3')

    db.store_run(b'\x02*', b'\x03')
    db.store_run(b'\x02*', b'\x42\x43')
