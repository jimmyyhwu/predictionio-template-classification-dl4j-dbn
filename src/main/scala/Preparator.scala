package org.template.classification

import io.prediction.controller.PPreparator
import io.prediction.data.storage.Event

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.nd4j.linalg.dataset.DataSet

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(records = trainingData.records)
  }
}

class PreparedData(
  val records: DataSet
) extends Serializable
