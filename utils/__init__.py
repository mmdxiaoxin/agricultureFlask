from model import AgriMenu


def build_nested_menu(menu_items, parent_id=None):
    nested_menu = []
    seen_paths = set()
    for item in menu_items:
        if item.parent_id == parent_id:
            if item.path in seen_paths:
                continue
            seen_paths.add(item.path)
            menu_item_dict = {
                "path": item.path,
                "name": item.name,
                "component": item.component,
                "meta": {
                    "icon": item.icon,
                    "title": item.title,
                    "isLink": "",
                    "isHide": item.isHide,
                    "isFull": item.isFull,
                    "isAffix": item.isAffix,
                    "isKeepAlive": item.isKeepAlive
                }
            }
            # 递归构建子菜单
            children = build_nested_menu(menu_items, item.id)
            if children:
                menu_item_dict["children"] = children
            nested_menu.append(menu_item_dict)
    return nested_menu


# 查询数据库中的菜单数据并构建嵌套菜单
def query_and_build_nested_menu():
    menu_items = AgriMenu.query.all()
    nested_menu = build_nested_menu(menu_items)
    return nested_menu
