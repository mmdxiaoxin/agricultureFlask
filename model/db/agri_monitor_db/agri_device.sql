create table agri_device
(
    id          int auto_increment
        primary key,
    site_id     int                     not null,
    device_name varchar(45)             null,
    business_id varchar(45)             null,
    device_id   varchar(45)             null,
    equipment   varchar(45)             null,
    version     varchar(45)             null,
    api         varchar(150)            null,
    collect_run varchar(10) default '0' null,
    constraint agri_device_agri_site_id_fk
        foreign key (site_id) references agri_site (id)
);

