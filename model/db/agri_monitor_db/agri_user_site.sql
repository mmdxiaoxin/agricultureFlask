create table agri_user_site
(
    id      int auto_increment
        primary key,
    user_id int not null,
    site_id int not null,
    constraint agri_user_site_agri_site_id_fk
        foreign key (site_id) references agri_site (id),
    constraint agri_user_site_agri_user_id_fk
        foreign key (user_id) references agri_user (id)
);

