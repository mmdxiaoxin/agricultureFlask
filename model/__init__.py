from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(50))

    def __repr__(self):
        return f'<User {self.username}>'


class Device(db.Model):
    __tablename__ = 'device'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    address_id = db.Column(db.Integer, db.ForeignKey('address.id'))
    device_name = db.Column(db.String(45))
    business_id = db.Column(db.String(45))
    device_id = db.Column(db.String(45))
    equipment = db.Column(db.String(45))
    version = db.Column(db.String(45))
    api = db.Column(db.String(150))
    database_name = db.Column(db.String(45))
    collect_run = db.Column(db.String(10), default='0')

    # 定义与 Address 表的关联关系
    address = db.relationship('Address', back_populates='devices')


# 如果有外键关联的地址表，也需要定义 Address 模型类
class Address(db.Model):
    __tablename__ = 'address'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(45))

    # 定义关联到 Device 表的反向引用关系
    devices = db.relationship('Device', back_populates='address')
