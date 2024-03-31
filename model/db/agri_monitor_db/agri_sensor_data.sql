create table agri_sensor_data
(
    id         int auto_increment
        primary key,
    device_id  int                                 not null,
    AD1        float                               null,
    AL1        float                               null,
    AF1        float                               null,
    AE1        float                               null,
    AB1        float                               null,
    AA1        float                               null,
    AH1        float                               null,
    AI1        float                               null,
    AC1        int                                 null,
    AJ1        int                                 null,
    BD1        int                                 null,
    createTime timestamp default CURRENT_TIMESTAMP not null,
    constraint agri_sensor_data_ibfk_1
        foreign key (device_id) references agri_device (id),
    constraint sensor_data_check
        check ((`AD1` is not null) or (`AL1` is not null) or (`AF1` is not null) or (`AE1` is not null) or
               (`AB1` is not null) or (`AA1` is not null) or (`AH1` is not null) or (`AI1` is not null) or
               (`AC1` is not null) or (`AJ1` is not null))
);

create index device_id
    on agri_sensor_data (device_id);

