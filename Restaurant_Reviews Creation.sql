-- Table: "Yelp"."Restaurant_Reviews"

-- DROP TABLE "Yelp"."Restaurant_Reviews";

CREATE TABLE yelp.reviews
(
    Name character varying COLLATE pg_catalog."default",
	ZipCode numeric,
    Date date,
    Review_Count numeric,
	Review character varying COLLATE pg_catalog."default",
	Review_Vader_Comp numeric,
	id SERIAL PRIMARY KEY
)

TABLESPACE pg_default;

ALTER TABLE yelp.reviews
    OWNER to postgres;