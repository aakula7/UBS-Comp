-- Table: "Yelp"."Restaurant_Information"

-- DROP TABLE "Yelp"."Restaurant_Information";

CREATE TABLE yelp.attributes
(
    "Name" character varying COLLATE pg_catalog."default",
	"ZipCode" numeric,
    "Dollars" text COLLATE pg_catalog."default",
    "Photos" character varying COLLATE pg_catalog."default",
	"Staff wears masks" numeric,
	"Staff wears gloves" numeric,
	"Contactless payments" numeric,
	"In-person visits" numeric,
    "Sit-down dining" numeric,
    "Masks required" numeric,
    "Curbside pickup" numeric,
    "Limited capacity" numeric,
    "Social distancing enforced" numeric,
    "Sanitizing between customers" numeric,
    "Open to All" numeric,
	"Dairy-Free Options" numeric,
    "Outdoor seating" numeric,
    "Virtual consultations" numeric,
	"Shipping" numeric,
	"Virtual tasting sessions" numeric,
	"Mobile services" numeric,
	"Accepts Credit Cards" numeric,
    "Hand sanitizer provided" numeric,
    "Takes Reservations" numeric,
    "Temperature checks" numeric,
    "Remote services" numeric,
    "Health Score" numeric,
    "Virtual tours" numeric,
	"Curbside drop-off" numeric,
    "Virtual experiences" numeric,
    "Virtual performances" numeric,
    "Drive-thru" numeric,
	"In-store shopping" numeric,
    "Gift cards" numeric,
    "Casual Dress" numeric,
    "Takeout" numeric,
    "Delivery" numeric,
	"id" SERIAL PRIMARY KEY
)

TABLESPACE pg_default;

ALTER TABLE yelp.attributes
    OWNER to postgres;