CREATE TABLE `customer` (
    `c_custkey` BIGINT NOT NULL,
    `c_name` VARCHAR(30),
    `c_address` VARCHAR(30),
    `c_city` CHAR(20),
    `c_nation` CHAR(20),
    `c_region` CHAR(20),
    `c_phone` CHAR(20),
    `c_mktsegment` CHAR(20),
    PRIMARY KEY (`c_custkey`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `date` (
    `d_datekey` BIGINT NOT NULL,
    `d_date` CHAR(20),
    `d_dayofweek` CHAR(10),
    `d_month` CHAR(10),
    `d_year` BIGINT,
    `d_yearmonthnum` BIGINT,
    `d_yearmonth` CHAR(10),
    `d_daynuminmonth` BIGINT,
    `d_daynuminyear` BIGINT,
    `d_monthnuminyear` BIGINT,
    `d_weeknuminyear` BIGINT,
    `d_sellingseason` CHAR(20),
    `d_lastdayinweekfl` BIGINT,
    `d_lastdayinmonthfl` BIGINT,
    `d_holidayfl` BIGINT,
    `d_weekdayfl` BIGINT,
    PRIMARY KEY (`d_datekey`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `lineorder` (
    `lo_orderkey` BIGINT NOT NULL,
    `lo_linenumber` BIGINT NOT NULL,
    `lo_custkey` BIGINT,
    `lo_partkey` BIGINT,
    `lo_suppkey` BIGINT,
    `lo_orderdate` BIGINT,
    `lo_orderpriority` CHAR(20),
    `lo_shippriority` CHAR(1),
    `lo_quantity` BIGINT,
    `lo_extendedprice` BIGINT,
    `lo_ordtotalprice` BIGINT,
    `lo_discount` BIGINT,
    `lo_revenue` BIGINT,
    `lo_supplycost` BIGINT,
    `lo_tax` BIGINT,
    `lo_commitdate` BIGINT,
    `lo_shipmode` CHAR(10),
    PRIMARY KEY (`lo_orderkey`, `lo_linenumber`),
    KEY `idx_custkey` (`lo_custkey`),
    KEY `idx_partkey` (`lo_partkey`),
    KEY `idx_suppkey` (`lo_suppkey`),
    KEY `idx_orderdate` (`lo_orderdate`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `part` (
    `p_partkey` BIGINT NOT NULL,
    `p_name` VARCHAR(30),
    `p_mfgr` CHAR(10),
    `p_category` CHAR(10),
    `p_brand1` CHAR(10),
    `p_color` VARCHAR(20),
    `p_type` VARCHAR(30),
    `p_size` BIGINT,
    `p_container` CHAR(10),
    PRIMARY KEY (`p_partkey`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `supplier` (
    `s_suppkey` BIGINT NOT NULL,
    `s_name` CHAR(30),
    `s_address` VARCHAR(30),
    `s_city` CHAR(20),
    `s_nation` CHAR(20),
    `s_region` CHAR(20),
    `s_phone` CHAR(20),
    PRIMARY KEY (`s_suppkey`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;