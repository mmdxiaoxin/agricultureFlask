from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class AgriAuth(db.Model):
    id = db.Column(db.Integer, primary_key=True)


class AgriDevice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    site_id = db.Column(db.Integer, db.ForeignKey('agri_site.id'), nullable=False)
    device_name = db.Column(db.String(45))
    business_id = db.Column(db.String(45))
    device_id = db.Column(db.String(45))
    equipment = db.Column(db.String(45))
    version = db.Column(db.String(45))
    api = db.Column(db.String(150))
    collect_run = db.Column(db.String(10), default='0')

    # 定义与 AgriSite 表的反向引用关系
    site = db.relationship('AgriSite', back_populates='devices')


class AgriMenu(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    component = db.Column(db.String(255), nullable=False)
    icon = db.Column(db.String(255))
    title = db.Column(db.String(255), nullable=False)
    isLink = db.Column(db.Boolean, nullable=False)
    isHide = db.Column(db.Boolean, nullable=False)
    isFull = db.Column(db.Boolean, nullable=False)
    isAffix = db.Column(db.Boolean, nullable=False)
    isKeepAlive = db.Column(db.Boolean, nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('agri_menu.id'))


class AgriSensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, db.ForeignKey('agri_device.id'), nullable=False)
    AD1 = db.Column(db.Float)
    AL1 = db.Column(db.Float)
    AF1 = db.Column(db.Float)
    AE1 = db.Column(db.Float)
    AB1 = db.Column(db.Float)
    AA1 = db.Column(db.Float)
    AH1 = db.Column(db.Float)
    AI1 = db.Column(db.Float)
    AC1 = db.Column(db.Integer)
    AJ1 = db.Column(db.Integer)
    BD1 = db.Column(db.Integer)
    createTime = db.Column(db.TIMESTAMP, nullable=False, default=db.func.current_timestamp())

    def to_dict(self, columns):
        return {column: getattr(self, column) for column in columns if column != 'device_id'}


class AgriSite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    site_name = db.Column(db.String(100), nullable=False)

    # 定义与 AgriDevice 表的反向引用关系
    devices = db.relationship('AgriDevice', back_populates='site')


class AgriUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(255))


class AgriUserMenu(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('agri_user.id'), nullable=False)
    menu_id = db.Column(db.Integer, db.ForeignKey('agri_menu.id'), nullable=False)


class AgriUserSite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('agri_user.id'), nullable=False)
    site_id = db.Column(db.Integer, db.ForeignKey('agri_site.id'), nullable=False)
