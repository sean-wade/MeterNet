import sqlite3


class FaceSQL(object):
    def __init__(self, db_path="./face.db"):
        #连接到数据库
        #如果数据库不存在的话，将会自动创建一个 数据库
        self.conn = sqlite3.connect(db_path, check_same_thread = False)

        #创建一个游标 curson
        self.cursor = self.conn.cursor()

        #创建meter表,  如果不存在的话
        sql = 'CREATE TABLE IF NOT EXISTS  Meter(id integer primary key, name varchar(100), identifyID varchar(50000))'
        self.cursor.execute(sql)

    
    def insert(self, idx, name, identifyID):
        try:
            self.cursor.execute("insert into Meter values(?,?,?)",(idx, name, identifyID))
            self.conn.commit()
        except Exception as e:
            print(e)


    def queryAll(self):
        self.cursor.execute("select * from Meter")  # where id > 0
        return self.cursor.fetchall()  


    def queryID(self, qID):
        self.cursor.execute("select * from Meter where id = %d"%qID)  # where id > 0
        return self.cursor.fetchall()


    def deleteName(self, name):
        self.cursor.execute("delete from Meter where name = '%s'" % name)
        self.conn.commit()
        return self.queryAll()


    def close(self):
        self.cursor.close()
        self.conn.close()


if __name__ == "__main__":
    faceDB = FaceSQL()
    #faceDB.insert(1, 'led', "1.22,3.44,4.55")
    faceDB.insert(2, 'pointer', "11.22,37.11,4.55")
    faceDB.insert(3, 'switch', "12.22,88.33,4.55")
    print(faceDB.queryAll())