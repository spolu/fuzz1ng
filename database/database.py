import base64
import sqlite3
import xxhash


class Database:
    """ `Database` stores coverage pathese and inputs.

    `Database` is an sqlite3 backed database that associates inputs (binary)
    with coverage path hashes.
    """
    def __init__(
            self,
            db_path: str,
    ) -> None:
        self._conn = sqlite3.connect(db_path)

        c = self._conn.cursor()
        c.execute('PRAGMA encoding="UTF-8";')

        c.execute('''
            CREATE TABLE IF NOT EXISTS runs(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              input_hash TEXT,
              path_hash TEXT,
              raw_input BLOB
            );
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS runs_path_hash_idx ON runs(path_hash);
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS runs_input_hash_idx ON runs(input_hash);
        ''')
        self._conn.commit()

    def store_run(
            self,
            path: bytes,
            input: bytes,
    ) -> None:
        path_hash = base64.b64encode(path)
        x = xxhash.xxh64()
        x.update(input)
        input_hash = base64.b64encode(x.digest())

        c = self._conn.cursor()

        c.execute('''
            SELECT id, path_hash FROM runs WHERE input_hash = ?
        ''', (input_hash,))
        run = c.fetchone()
        if run is None:
            c.execute('''
                INSERT INTO runs(input_hash, path_hash, raw_input)
                  VALUES(?, ?, ?)
            ''', (input_hash, path_hash, input))
        else:
            assert run[1] == path_hash

        self._conn.commit()
