create table agri_menu
(
    id          int auto_increment
        primary key,
    path        varchar(255) not null,
    name        varchar(255) not null,
    component   varchar(255) not null,
    icon        varchar(255) null,
    title       varchar(255) not null,
    isLink      tinyint(1)   not null,
    isHide      tinyint(1)   not null,
    isFull      tinyint(1)   not null,
    isAffix     tinyint(1)   not null,
    isKeepAlive tinyint(1)   not null,
    parent_id   int          null,
    constraint agri_menu_ibfk_1
        foreign key (parent_id) references agri_menu (id)
);

create index parent_id
    on agri_menu (parent_id);

