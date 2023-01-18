import numpy as np
import pandas as pd

from math import radians, cos, sin, asin, sqrt

from time import time

def haversine_distance(lat1, lng1, lat2, lng2, degrees=True):
    """La formule de Havrsine permet de determiner la distance sous forme de cercle (distance géodisique) 
        entre deux points d'une sphère, à partie de leurs longitudes and latitudes.

    Paramètres
    ----------
    lat1, lat2 : float
        Latitudes deux 2 points. 
    lng1, lng2 : float
        Longitudes de 2 points.
    degrees : boolean
        Si c'est True, effectue la conversion des radians en degrés.
    """
    
    r = 3956 #Rayon de la planète terre en miles
    
    if degrees:
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    
    #Formule de Haversine
    dlng = lng2 - lng1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    d = 2 * r * asin(sqrt(a))  

    return d

def refaire_dataset(dpath = 'datas/', initial = False, period = 2):
    """Nettoyage et feature engineering sur les différents jeux de données d'Olist
        pour la préparation de la classification non supervisée selon le modèle du K-Means.
        
        Paramètres
    ----------
    dpath : str
        Chemin vers le dossier où se trouvent les jeu de données.
    initial : booléen
        Définit si le jeu de données créé est le jeu de données initial.
    period : int
        Période d'incrémentation en mois après le jeu de données initial.
    """
    start_time = time()
    print("Création du dataset en cours ...")
    
    #Chemin
    root_path = dpath
    
    #Import des données

    df_customers = pd.read_csv(root_path + 'olist_customers_dataset.csv')
    df_geolocation = pd.read_csv(root_path + 'olist_geolocation_dataset.csv')
    df_items = pd.read_csv(root_path + 'olist_order_items_dataset.csv')
    df_payments = pd.read_csv(root_path + 'olist_order_payments_dataset.csv')
    df_reviews = pd.read_csv(root_path + 'olist_order_reviews_dataset.csv')
    df_orders = pd.read_csv(root_path + 'olist_orders_dataset.csv')
    df_products = pd.read_csv(root_path + 'olist_products_dataset.csv')
    df_sellers = pd.read_csv(root_path + 'olist_sellers_dataset.csv')
    df_category = pd.read_csv(root_path + 'olist_product_category_name_translation.csv')

    #Localisation
    df_geolocation = df_geolocation.groupby(["geolocation_state"]).agg({
                "geolocation_lat": "mean",
                "geolocation_lng": "mean"})
    
    #Concaténation
    #Commande
    df_orders.drop(["order_approved_at", "order_delivered_carrier_date", "order_estimated_delivery_date"], axis = 1, inplace = True)

    df_items.drop(["seller_id", "shipping_limit_date"], axis = 1, inplace = True)
    
    df_items = pd.merge(df_items, df_orders, how = "left", on = "order_id")
    
    datetime_cols = ["order_purchase_timestamp", "order_delivered_customer_date"]
    for col in datetime_cols:
        df_items[col] = df_items[col].astype('datetime64[ns]')
        
    #Commandes mensuelles
    df_items["sale_month"] = df_items['order_purchase_timestamp'].dt.month
    
    #Sélection de commandes sur une période
    start = df_items["order_purchase_timestamp"].min()
    
    if(initial == True):
        period = 12
    else:
        period = 12 + period

    stop = start + pd.DateOffset(months = period)
        
    df_items = df_items[(df_items["order_purchase_timestamp"] >= start) & (df_items["order_purchase_timestamp"] < stop)]
    
    #Liste des commandes sur une période donnée
    period_orders = df_items.order_id.unique()
    
    #Calcule des autres indicateurs (features) sur une période donnée 
    df_payments = df_payments[df_payments["order_id"].isin(period_orders)]

    df_items = pd.merge(df_items, df_payments.groupby(by="order_id").agg({"payment_sequential": 'count', "payment_installments": 'sum'}), how = "left", on = "order_id")

    df_items = df_items.rename(columns = {"payment_sequential": "nb_payment_sequential", "payment_installments": "sum_payment_installments"})
    
    df_reviews = df_reviews[df_reviews["order_id"].isin(period_orders)]

    df_items = pd.merge(df_items, df_reviews.groupby("order_id").agg({"review_score": "mean"}), how = "left", on = "order_id")
    
    #Temps de livraison
    df_items["delivery_delta_days"] = (df_items.order_delivered_customer_date - df_items.order_purchase_timestamp).dt.round('1d').dt.days
    
    df_items.drop("order_delivered_customer_date", axis = 1, inplace = True)
    
    #Produits
    df_products = pd.merge(df_products, df_category, how = "left", on = "product_category_name")
    
    variables_inutiles = ["product_category_name", "product_weight_g",
                         "product_length_cm", "product_height_cm",
                         "product_width_cm", "product_name_lenght", 
                         "product_description_lenght", "product_photos_qty"]
    
    df_products.drop(variables_inutiles, axis = 1, inplace = True)
    
    df_products = df_products.rename(columns = {"product_category_name_english": "product_category_name"})
    
    df_products['product_category'] = np.where((df_products['product_category_name'].str.contains("fashio|luggage") == True),
                                    'fashion_clothing_accessories',
                            np.where((df_products['product_category_name'].str.contains("health|beauty|perfum") == True),
                                     'health_beauty',
                            np.where((df_products['product_category_name'].str.contains("toy|baby|diaper") == True),
                                     'toys_baby',
                            np.where((df_products['product_category_name'].str.contains("book|cd|dvd|media") == True),
                                     'books_cds_media',
                            np.where((df_products['product_category_name'].str.contains("grocer|food|drink") == True), 
                                     'groceries_food_drink',
                            np.where((df_products['product_category_name'].str.contains("phon|compu|tablet|electro|consol") == True), 
                                     'technology',
                            np.where((df_products['product_category_name'].str.contains("home|furnitur|garden|bath|house|applianc") == True), 
                                     'home_furniture',
                            np.where((df_products['product_category_name'].str.contains("flow|gift|stuff") == True),
                                     'flowers_gifts',
                            np.where((df_products['product_category_name'].str.contains("sport") == True),
                                     'sport',
                                     'other')))))))))
    
    df_products.drop("product_category_name", axis = 1, inplace = True)
    
    df_items = pd.merge(df_items, df_products, how = "left", on = "product_id")
    
    #Encodage des catégories des colonnes
    df_items = pd.get_dummies(df_items, columns = ["product_category"], prefix = "", prefix_sep = "")
    
    #Clients
    df_items = pd.merge(df_items, df_customers[["customer_id", "customer_unique_id", "customer_state"]], on = "customer_id", how = "left")
    
    #Regroupement des données pour chaque client unique
    data = df_items.groupby(["customer_unique_id"]).agg(
        nb_orders = pd.NamedAgg(column = "order_id", aggfunc = "nunique"),
        total_items = pd.NamedAgg(column = "order_item_id", aggfunc = "count"),
        total_spend = pd.NamedAgg(column = "price", aggfunc = "sum"),
        total_freight = pd.NamedAgg(column = "freight_value", aggfunc = "sum"),
        mean_payment_sequential = pd.NamedAgg(column = "nb_payment_sequential", aggfunc = "mean"),
        mean_payment_installments = pd.NamedAgg(column = "sum_payment_installments", aggfunc = "mean"),
        mean_review_score = pd.NamedAgg(column = "review_score", aggfunc = "mean"),
        mean_delivery_days = pd.NamedAgg(column = "delivery_delta_days", aggfunc = "mean"),
        books_cds_media = pd.NamedAgg(column = "books_cds_media", aggfunc = "sum"),
        fashion_clothing_accessories = pd.NamedAgg(column = "fashion_clothing_accessories", aggfunc = "sum"),
        flowers_gifts=pd.NamedAgg(column = "flowers_gifts", aggfunc = "sum"),
        groceries_food_drink = pd.NamedAgg(column = "groceries_food_drink", aggfunc = "sum"),
        health_beauty = pd.NamedAgg(column = "health_beauty", aggfunc = "sum"),
        home_furniture = pd.NamedAgg(column = "home_furniture", aggfunc = "sum"),
        other=pd.NamedAgg(column = "other", aggfunc = "sum"),
        sport=pd.NamedAgg(column = "sport", aggfunc = "sum"),
        technology=pd.NamedAgg(column = "technology", aggfunc = "sum"),
        toys_baby=pd.NamedAgg(column = "toys_baby", aggfunc = "sum"),
        customer_state=pd.NamedAgg(column = "customer_state", aggfunc = "max"),
        first_order=pd.NamedAgg(column = "order_purchase_timestamp", aggfunc = "min"),
        last_order=pd.NamedAgg(column = "order_purchase_timestamp", aggfunc = "max"),
        favorite_sale_month=pd.NamedAgg(column = "sale_month", aggfunc = lambda x:x.value_counts().index[0]))
    
    #Feature engineering final
    #Rapport des objets par catégorie
    cat_features = data.columns[7:17]
    for c in cat_features:
        data[c] = data[c] / data["total_items"]
    
    #Délai moyen entre deux commandes
    data["order_mean_delay"] = [(y[1] - y[0]).round('1d').days if y[1] != y[0]
                                else (stop - y[0]).round('1d').days
                                for x, y in data[["first_order","last_order"]].iterrows()]
    
    data["order_mean_delay"] = data["order_mean_delay"] / data["nb_orders"]
    
    data.drop(["first_order", "last_order"], axis = 1, inplace = True)
    
    #Rapport du prix du fret et prix total
    data["freight_ratio"] = (round(data["total_freight"] / (data["total_spend"] + data["total_freight"]), 2))
    
    data["total_spend"] = (data["total_spend"] + data["total_freight"])
    
    data.drop("total_freight", axis = 1, inplace = True)
    
    #Ajout de la distance de Haversine pour les états recensés dans le jeu de données
    #Distance de Haversine
    olist_lat = -25.43045
    olist_lon = -49.29207
    df_geolocation['haversine_distance'] = [haversine_distance(olist_lat, olist_lon, x, y)
                                         for x, y in zip(df_geolocation.geolocation_lat,
                                                         df_geolocation.geolocation_lng)]
    
    data = pd.merge(data.reset_index(), df_geolocation[["haversine_distance"]], how = "left", left_on = "customer_state", right_on = "geolocation_state")
    
    data.drop(["customer_state"], axis = 1, inplace = True)
    
    data.set_index("customer_unique_id", inplace = True)
    
    #Traitement des valeurs nulles
    valeur_remplie = data.isnull().sum()
    
    valeur_remplie = list(valeur_remplie[valeur_remplie.values > 0].index)
    
    print(54*"_")
    print("Indicateurs ou Features complétées avec la valeur la plus fréquente :")
    print(54*"_")
    for f in valeur_remplie:
        data[f] = data[f].fillna(data[f].mode()[0])
        print(f,"\t", data[f].mode()[0])
    print(54*"_")
        
    end_time = time()
    print("Durée d'execution du Feature engineering : {:.2f}s".format(end_time - start_time))

    return data