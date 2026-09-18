# wms.py
# WareBot AI - Warehouse Management System Simulation

import pandas as pd


# ============================================================
# 1. INVENTORY DATA
# ============================================================

INVENTORY = [
    {
        "product_id": "SKU001",
        "product_name": "Wireless Mouse",
        "location": "A01",
        "quantity": 120,
        "reorder_level": 30,
    },
    {
        "product_id": "SKU002",
        "product_name": "Keyboard",
        "location": "A02",
        "quantity": 75,
        "reorder_level": 25,
    },
    {
        "product_id": "SKU003",
        "product_name": "USB Cable",
        "location": "B01",
        "quantity": 18,
        "reorder_level": 25,
    },
    {
        "product_id": "SKU004",
        "product_name": "Webcam",
        "location": "B02",
        "quantity": 42,
        "reorder_level": 15,
    },
    {
        "product_id": "SKU005",
        "product_name": "Headset",
        "location": "C01",
        "quantity": 12,
        "reorder_level": 20,
    },
    {
        "product_id": "SKU006",
        "product_name": "Laptop Stand",
        "location": "C02",
        "quantity": 60,
        "reorder_level": 20,
    },
]


# ============================================================
# 2. ORDER DATA
# ============================================================

ORDERS = [
    {
        "order_id": "ORD1001",
        "product_id": "SKU001",
        "quantity": 5,
        "priority": "High",
        "status": "Pending",
    },
    {
        "order_id": "ORD1002",
        "product_id": "SKU002",
        "quantity": 8,
        "priority": "Medium",
        "status": "Pending",
    },
    {
        "order_id": "ORD1003",
        "product_id": "SKU003",
        "quantity": 6,
        "priority": "High",
        "status": "Pending",
    },
    {
        "order_id": "ORD1004",
        "product_id": "SKU004",
        "quantity": 4,
        "priority": "Low",
        "status": "Pending",
    },
    {
        "order_id": "ORD1005",
        "product_id": "SKU005",
        "quantity": 10,
        "priority": "High",
        "status": "Pending",
    },
]


# ============================================================
# 3. GET INVENTORY
# ============================================================

def get_inventory():

    return pd.DataFrame(INVENTORY)


# ============================================================
# 4. GET ORDERS
# ============================================================

def get_orders():

    return pd.DataFrame(ORDERS)


# ============================================================
# 5. CHECK STOCK
# ============================================================

def check_stock(product_id, required_quantity):

    for item in INVENTORY:

        if item["product_id"] == product_id:

            available = item["quantity"]

            return {
                "product_id": product_id,
                "available": available,
                "required": required_quantity,
                "sufficient": available >= required_quantity,
            }

    return {
        "product_id": product_id,
        "available": 0,
        "required": required_quantity,
        "sufficient": False,
    }


# ============================================================
# 6. LOW STOCK DETECTION
# ============================================================

def get_low_stock_items():

    low_stock = []

    for item in INVENTORY:

        if item["quantity"] <= item["reorder_level"]:

            low_stock.append(item)

    return pd.DataFrame(low_stock)


# ============================================================
# 7. ORDER PROCESSING
# ============================================================

def process_order(order_id):

    for order in ORDERS:

        if order["order_id"] == order_id:

            stock = check_stock(
                order["product_id"],
                order["quantity"]
            )

            if stock["sufficient"]:

                order["status"] = "Ready for Picking"

                return {
                    "order_id": order_id,
                    "status": "Ready for Picking",
                    "message": "Stock available. Order can be picked."
                }

            else:

                order["status"] = "Stock Shortage"

                return {
                    "order_id": order_id,
                    "status": "Stock Shortage",
                    "message": "Insufficient inventory."
                }

    return {
        "order_id": order_id,
        "status": "Not Found",
        "message": "Order does not exist."
    }


# ============================================================
# 8. INVENTORY SUMMARY
# ============================================================

def inventory_summary():

    total_products = len(INVENTORY)

    total_units = sum(
        item["quantity"]
        for item in INVENTORY
    )

    low_stock = len(
        get_low_stock_items()
    )

    return {
        "total_products": total_products,
        "total_units": total_units,
        "low_stock_items": low_stock,
    }


# ============================================================
# 9. DEMO
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("       WAREBOT WMS SYSTEM")
    print("===================================")

    print("\nInventory:")
    print(get_inventory())

    print("\nOrders:")
    print(get_orders())

    print("\nInventory Summary:")

    summary = inventory_summary()

    print(
        f"Total Products : {summary['total_products']}"
    )

    print(
        f"Total Units    : {summary['total_units']}"
    )

    print(
        f"Low Stock      : {summary['low_stock_items']}"
    )

    print("\nLow Stock Items:")

    print(get_low_stock_items())

    print("\nOrder Processing:")

    result = process_order("ORD1001")

    print(result)