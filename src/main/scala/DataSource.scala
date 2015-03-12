package org.template.classification

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.Event
import io.prediction.data.storage.Storage

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.nd4j.linalg.dataset.DataSet


case class DataSourceParams(appId: Int) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {
    val eventsDb = Storage.getPEvents()
    // read all events of EVENT involving ENTITY_TYPE and TARGET_ENTITY_TYPE
    /*val eventsRDD: RDD[Event] = eventsDb.find(
      appId = dsp.appId,
      entityType = Some("ENTITY_TYPE"),
      eventNames = Some(List("EVENT")),
      targetEntityType = Some(Some("TARGET_ENTITY_TYPE")))(sc)*/
    val iter: DataSetIterator = new IrisDataSetIterator(150, 150)
    val next: DataSet = iter.next(110)
    next.normalizeZeroMeanZeroUnitVariance


    new TrainingData(next)
  }
}

class TrainingData(
  val records: DataSet
) extends Serializable {
  override def toString = {
    s"events: [${records.numExamples()}] (${records.get(0)}...)"
  }
}
