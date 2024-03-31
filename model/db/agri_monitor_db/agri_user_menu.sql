create table agri_user_menu
(
    id      int auto_increment
        primary key,
    user_id int not null,
    menu_id int not null,
    constraint agri_user_menu_agri_menu_id_fk
        foreign key (menu_id) references agri_menu (id),
    constraint agri_user_menu_agri_user_id_fk
        foreign key (user_id) references agri_user (id)
);

