#!/usr/bin/env python3
"""
Helper module for SQL Injection attack dataset generation.

This module provides functions to generate datasets for SQL Injection attack detection.
"""

import os
import sys
import random
import pandas as pd
import numpy as np
from typing import List
from src.logger import dataset_generator_logger as logger

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# Sample field names for SQL queries
FIELD_NAMES = [
    "id",
    "user_id",
    "username",
    "email",
    "password",
    "name",
    "first_name",
    "last_name",
    "address",
    "phone",
    "status",
    "created_at",
    "updated_at",
    "product_id",
    "category_id",
    "order_id",
    "quantity",
    "price",
    "title",
    "description",
    "content",
    "image_url",
]

# Sample table names
TABLE_NAMES = [
    "users",
    "products",
    "orders",
    "categories",
    "customers",
    "payments",
    "blog_posts",
    "comments",
    "inventory",
    "shipping",
    "cart_items",
    "sessions",
    "roles",
    "permissions",
    "audit_logs",
    "settings",
    "profiles",
    "addresses",
]

# Sample operators for WHERE clauses
OPERATORS = [
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "<>",
    "LIKE",
    "IN",
    "BETWEEN",
    "IS NULL",
    "IS NOT NULL",
]

# Sample SQL functions
SQL_FUNCTIONS = [
    "COUNT",
    "SUM",
    "AVG",
    "MAX",
    "MIN",
    "CONCAT",
    "SUBSTRING",
    "UPPER",
    "LOWER",
    "DATE",
    "NOW",
]


def generate_normal_query() -> str:
    """
    Generate a normal SQL query.

    Returns:
        A string representing a normal SQL query
    """
    # Choose a query type
    query_type = random.choice(["SELECT", "INSERT", "UPDATE", "DELETE"])

    if query_type == "SELECT":
        # Generate SELECT query
        num_fields = random.randint(1, 5)
        if num_fields == 1 and random.random() < 0.3:
            # Use a SQL function
            fields = [f"{random.choice(SQL_FUNCTIONS)}({random.choice(FIELD_NAMES)})"]
        else:
            fields = random.sample(FIELD_NAMES, num_fields)

        table = random.choice(TABLE_NAMES)

        query = f"SELECT {', '.join(fields)} FROM {table}"

        # Add WHERE clause
        if random.random() < 0.8:
            field = random.choice(FIELD_NAMES)
            operator = random.choice(OPERATORS)

            if operator == "=":
                value = random.choice(
                    [
                        f"'{random.randint(1, 1000)}'",
                        f"'{random.choice(['active', 'inactive', 'pending', 'completed'])}'",
                    ]
                )
            elif operator in [">", "<", ">=", "<="]:
                value = str(random.randint(1, 1000))
            elif operator == "LIKE":
                value = f"'%{random.choice(['user', 'admin', 'test', 'product'])}%'"
            elif operator == "IN":
                num_values = random.randint(2, 4)
                values = [str(random.randint(1, 100)) for _ in range(num_values)]
                value = f"({', '.join(values)})"
            elif operator == "BETWEEN":
                value = f"{random.randint(1, 50)} AND {random.randint(51, 100)}"
            elif operator in ["IS NULL", "IS NOT NULL"]:
                value = ""
            else:
                value = f"'{random.randint(1, 1000)}'"

            query += f" WHERE {field} {operator}"
            if value:
                query += f" {value}"

        # Add ORDER BY
        if random.random() < 0.5:
            query += f" ORDER BY {random.choice(FIELD_NAMES)} {random.choice(['ASC', 'DESC'])}"

        # Add LIMIT
        if random.random() < 0.6:
            query += f" LIMIT {random.randint(1, 100)}"

            # Add OFFSET
            if random.random() < 0.3:
                query += f" OFFSET {random.randint(0, 50)}"

    elif query_type == "INSERT":
        # Generate INSERT query
        table = random.choice(TABLE_NAMES)

        num_fields = random.randint(2, 5)
        fields = random.sample(FIELD_NAMES, num_fields)

        values = []
        for _ in range(num_fields):
            value_type = random.choice(["string", "number", "null"])
            if value_type == "string":
                values.append(f"'value{random.randint(1, 100)}'")
            elif value_type == "number":
                values.append(str(random.randint(1, 1000)))
            else:
                values.append("NULL")

        query = (
            f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({', '.join(values)})"
        )

    elif query_type == "UPDATE":
        # Generate UPDATE query
        table = random.choice(TABLE_NAMES)

        num_fields = random.randint(1, 3)
        fields = random.sample(FIELD_NAMES, num_fields)

        set_clauses = []
        for field in fields:
            value_type = random.choice(["string", "number", "null"])
            if value_type == "string":
                set_clauses.append(f"{field} = 'value{random.randint(1, 100)}'")
            elif value_type == "number":
                set_clauses.append(f"{field} = {random.randint(1, 1000)}")
            else:
                set_clauses.append(f"{field} = NULL")

        query = f"UPDATE {table} SET {', '.join(set_clauses)}"

        # Add WHERE clause (almost always needed for UPDATE)
        field = random.choice(FIELD_NAMES)
        operator = random.choice(["=", ">", "<"])
        value = random.choice(
            [str(random.randint(1, 1000)), f"'value{random.randint(1, 100)}'"]
        )

        query += f" WHERE {field} {operator} {value}"

    else:  # DELETE
        # Generate DELETE query
        table = random.choice(TABLE_NAMES)

        query = f"DELETE FROM {table}"

        # Add WHERE clause (almost always needed for DELETE)
        field = random.choice(FIELD_NAMES)
        operator = random.choice(["=", ">", "<"])
        value = random.choice(
            [str(random.randint(1, 1000)), f"'value{random.randint(1, 100)}'"]
        )

        query += f" WHERE {field} {operator} {value}"

    return query


def generate_malicious_query(attack_patterns: List[str]) -> str:
    """
    Generate a malicious SQL query using SQL Injection attack patterns.

    Args:
        attack_patterns: List of SQL Injection attack patterns

    Returns:
        A string representing a malicious SQL query
    """
    # Choose a random attack pattern
    attack_pattern = random.choice(attack_patterns)

    # There are several ways to construct an SQL injection
    attack_type = random.randint(1, 4)

    if attack_type == 1:
        # Modify a normal SELECT query
        query = f"SELECT * FROM users WHERE username = '{attack_pattern}'"
    elif attack_type == 2:
        # Modify a normal LOGIN query
        query = f"SELECT * FROM users WHERE username = 'admin' AND password = '{attack_pattern}'"
    elif attack_type == 3:
        # Modify a WHERE clause
        table = random.choice(TABLE_NAMES)
        field = random.choice(FIELD_NAMES)
        query = f"SELECT * FROM {table} WHERE {field} = {attack_pattern}"
    else:
        # Modify an INSERT or UPDATE statement
        table = random.choice(TABLE_NAMES)
        field = random.choice(FIELD_NAMES)
        if random.choice([True, False]):
            # INSERT
            query = f"INSERT INTO {table} ({field}) VALUES ('{attack_pattern}')"
        else:
            # UPDATE
            query = f"UPDATE {table} SET {field} = '{attack_pattern}' WHERE id = 1"

    return query


def generate_mixed_dataset(
    attack_patterns: List[str],
    num_normal: int = 2500,
    num_malicious: int = 2500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a mixed dataset of normal and malicious SQL queries.

    Args:
        attack_patterns: List of SQL Injection attack patterns
        num_normal: Number of normal samples to generate
        num_malicious: Number of malicious samples to generate
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame containing the mixed dataset
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Generate normal queries
    normal_queries = []
    for _ in range(num_normal):
        query = generate_normal_query()
        normal_queries.append({"query": query, "label": 0})

    # Generate malicious queries
    malicious_queries = []
    for _ in range(num_malicious):
        query = generate_malicious_query(attack_patterns)
        malicious_queries.append({"query": query, "label": 1})

    # Combine and shuffle the dataset
    dataset = pd.DataFrame(normal_queries + malicious_queries)
    dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(
        f"Generated SQL Injection dataset with {num_normal} normal and {num_malicious} malicious samples"
    )

    return dataset
