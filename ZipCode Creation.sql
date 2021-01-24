CREATE TABLE yelp.zipcode
(
    ZipCode bigint,
	PO_Name character varying COLLATE pg_catalog."default",
	State character varying COLLATE pg_catalog."default",
	County character varying COLLATE pg_catalog."default",
	geometry geometry
)

TABLESPACE pg_default;

ALTER TABLE yelp.zipcode
    OWNER to postgres;



