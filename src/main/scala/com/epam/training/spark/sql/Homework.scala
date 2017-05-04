package com.epam.training.spark.sql

import java.sql.Date
import java.time.LocalDate

import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.{SparkConf, SparkContext}

object Homework {
  val DELIMITER = ";"
  val RAW_BUDAPEST_DATA = "data/budapest_daily_1901-2010.csv"
  val OUTPUT_DUR = "output"

  def main(args: Array[String]): Unit = {
    val sparkConf: SparkConf = new SparkConf()
      .setAppName("EPAM BigData training Spark SQL homework")
      .setIfMissing("spark.master", "local[2]")
      .setIfMissing("spark.sql.shuffle.partitions", "10")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new HiveContext(sc)

    processData(sqlContext)

    sc.stop()

  }

  def processData(sqlContext: HiveContext): Unit = {

    /**
      * Task 1
      * Read csv data with DataSource API from provided file
      * Hint: schema is in the Constants object
      */
    val climateDataFrame: DataFrame = readCsvData(sqlContext, Homework.RAW_BUDAPEST_DATA)

    /**
      * Task 2
      * Find errors or missing values in the data
      * Hint: try to use udf for the null check
      */
    val errors: Array[Row] = findErrors(climateDataFrame)
    println(errors)

    /**
      * Task 3
      * List average temperature for a given day in every year
      */
    val averageTemeperatureDataFrame: DataFrame = averageTemperature(climateDataFrame, 1, 2)

    /**
      * Task 4
      * Predict temperature based on mean temperature for every year including 1 day before and after
      * For the given month 1 and day 2 (2nd January) include days 1st January and 3rd January in the calculation
      * Hint: if the dataframe contains a single row with a single double value you can get the double like this "df.first().getDouble(0)"
      */
    val predictedTemperature: Double = predictTemperature(climateDataFrame, 1, 2)
    println(s"Predicted temperature: $predictedTemperature")

  }

  def readCsvData(sqlContext: HiveContext, rawDataPath: String): DataFrame = {
    sqlContext.read
      .option("header", "true")
      .option("delimiter", DELIMITER)
      .schema(Constants.CLIMATE_TYPE)
      .csv(rawDataPath)
  }

  def findErrors(climateDataFrame: DataFrame): Array[Row] = {
    import climateDataFrame.sqlContext.implicits._

    climateDataFrame
      .agg(
        sum(when($"observation_date".isNull, 1).otherwise(0)),
        sum(when($"mean_temperature".isNull, 1).otherwise(0)),
        sum(when($"max_temperature".isNull, 1).otherwise(0)),
        sum(when($"min_temperature".isNull, 1).otherwise(0)),
        sum(when($"precipitation_mm".isNull, 1).otherwise(0)),
        sum(when($"precipitation_type".isNull, 1).otherwise(0)),
        sum(when($"sunshine_hours".isNull, 1).otherwise(0)))
      .collect()
  }

  def averageTemperature(climateDataFrame: DataFrame, monthNumber: Int, dayOfMonth: Int): DataFrame = {
    import climateDataFrame.sqlContext.implicits._

    val dateFilter: Date => Boolean =
      d =>
        d.toLocalDate.getMonthValue() == monthNumber &&
        d.toLocalDate.getDayOfMonth() == dayOfMonth

    val dateFilterUdf = udf(dateFilter)

    climateDataFrame
      .filter(dateFilterUdf($"observation_date"))
      .select($"mean_temperature")
  }

  def dateToTuple(date: LocalDate): (Int, Int) = (date.getMonthValue, date.getDayOfMonth)

  def datesInRange(month: Int, dayOfMonth: Int): List[(Int, Int)] = {
    val date = LocalDate.of(2017, month, dayOfMonth)
    List(dateToTuple(date.minusDays(1)), dateToTuple(date), dateToTuple(date.plusDays(1)))
  }

  def predictTemperature(climateDataFrame: DataFrame, monthNumber: Int, dayOfMonth: Int): Double = {
    import climateDataFrame.sqlContext.implicits._

    val dates = datesInRange(monthNumber, dayOfMonth)
    val dateFilter: Date => Boolean =
      d => {
        val localDate = d.toLocalDate()
        dates.contains(localDate.getMonthValue, localDate.getDayOfMonth())
    }

    val dateFilterUdf = udf(dateFilter)

    climateDataFrame
      .filter(dateFilterUdf($"observation_date"))
      .agg(avg($"mean_temperature"))
      .collectAsList()
      .get(0)
      .getDouble(0)
  }


}


